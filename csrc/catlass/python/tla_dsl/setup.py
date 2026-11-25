from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    from setuptools.command.editable_wheel import editable_wheel
except ImportError:

    class editable_wheel:
        pass


PROJECT_ROOT = Path(__file__).resolve().parent


def generate_tla_bindings() -> None:
    script = PROJECT_ROOT / "tools" / "generate_tla_python_bindings.py"
    if not script.is_file():
        raise RuntimeError(f"TableGen binding generator not found: {script}")

    print("==> Generating Tla Python op bindings")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _detect_compiler() -> dict[str, str]:
    """自动检测 clang/clang++，若版本为 19.1.7 则设置 CMAKE_C/CXX_COMPILER。

    用户可通过 CC/CXX 环境变量显式覆盖检测结果。
    若仅检测到 GCC 或无 clang，打印 warning 但不阻塞构建。
    """
    env: dict[str, str] = {}

    cc = os.environ.get("CC")
    cxx = os.environ.get("CXX")

    if cc and cxx:
        env["CMAKE_C_COMPILER"] = cc
        env["CMAKE_CXX_COMPILER"] = cxx
        return env

    clang = shutil.which("clang-19") or shutil.which("clang")
    clangxx = shutil.which("clang++-19") or shutil.which("clang++")

    if clang and clangxx:
        proc = subprocess.run(
            [clang, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"clang version (\S+)", proc.stdout)
        version = match.group(1) if match else "<unknown>"

        if version.startswith("19."):
            env["CMAKE_C_COMPILER"] = cc or clang
            env["CMAKE_CXX_COMPILER"] = cxx or clangxx
        else:
            print(
                f"WARNING: clang version is {version}, expected 19.x. "
                "While GCC-based builds may work in practice, clang is the "
                "recommended compiler for TLA / MLIR development.",
                file=sys.stderr,
            )
    else:
        print(
            "WARNING: clang/clang++ not found in PATH. "
            "While GCC-based builds may work in practice, clang is the "
            "recommended compiler for TLA / MLIR development.",
            file=sys.stderr,
        )

    return env


class CMakeExtension(Extension):
    """A setuptools extension whose build is delegated to CMake."""

    def __init__(
        self,
        name: str,
        *,
        sourcedir: str,
        target: str,
        cmake_output_dir: str,
    ) -> None:
        super().__init__(name=name, sources=[])
        self.sourcedir = sourcedir
        self.cmake_target = target
        self.cmake_output_dir = cmake_output_dir


class CMakeBuild(build_ext):
    """Invoke CMake instead of the setuptools compiler abstraction."""

    def run(self) -> None:
        if shutil.which("cmake") is None:
            raise RuntimeError("CMake was not found. Install CMake or add it to PATH.")

        generate_tla_bindings()
        super().run()

    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        source_dir = (PROJECT_ROOT / ext.sourcedir).resolve()
        extension_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        extension_dir = extension_path.parent
        extension_dir.mkdir(parents=True, exist_ok=True)

        build_type = os.environ.get(
            "CMAKE_BUILD_TYPE",
            "Debug" if self.debug else "Release",
        )
        generator = os.environ.get("CMAKE_GENERATOR")
        if not generator and shutil.which("ninja"):
            generator = "Ninja"
        extra_cmake_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))

        build_dir = self._get_build_directory(
            build_type=build_type,
            generator=generator,
            extra_cmake_args=extra_cmake_args,
        )
        build_dir.mkdir(parents=True, exist_ok=True)

        configure_command = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            # 使用宿主系统根目录作为 sysroot（容器/交叉构建时的关键护栏）。
            # 放在 CMAKE_ARGS 之前追加，用户可通过 CMAKE_ARGS 显式覆盖。
            "-DCMAKE_SYSROOT=/",
        ]

        compiler_env = _detect_compiler()
        for key, value in compiler_env.items():
            configure_command.append(f"-D{key}={value}")

        if generator:
            configure_command.extend(["-G", generator])
        configure_command.extend(extra_cmake_args)

        subprocess.run(
            configure_command,
            cwd=PROJECT_ROOT,
            check=True,
        )

        build_command = [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            build_type,
            "--target",
            ext.cmake_target,
        ]
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            jobs = self.parallel or os.cpu_count() or 1
            build_command.extend(["--parallel", str(jobs)])

        subprocess.run(
            build_command,
            cwd=PROJECT_ROOT,
            check=True,
        )

        cmake_out_dir = (build_dir / ext.cmake_output_dir).resolve()
        produced = sorted(cmake_out_dir.glob("_tla_type_bridge_native*.so"))

        if not produced:
            generated_files = "\n".join(
                f"  {path.name}" for path in sorted(cmake_out_dir.glob("*"))
            )
            raise RuntimeError(
                "CMake completed, but the expected Python extension "
                "was not generated:\n"
                f"  expected pattern: {cmake_out_dir}/_tla_type_bridge_native*.so\n"
                f"  output directory: {cmake_out_dir}\n"
                f"  generated files:\n{generated_files or '  <none>'}"
            )

        if self.inplace:
            if extension_path.is_symlink() or extension_path.exists():
                extension_path.unlink()
            os.symlink(produced[0], extension_path)
        else:
            shutil.copy2(produced[0], extension_path)

        if not extension_path.is_file():
            raise RuntimeError(
                f"Failed to relocate CMake extension to {extension_path}"
            )

    def _get_build_directory(
        self,
        *,
        build_type: str,
        generator: str | None,
        extra_cmake_args: list[str],
    ) -> Path:
        explicit_build_dir = os.environ.get("CMAKE_BUILD_DIR")
        if explicit_build_dir:
            path = Path(explicit_build_dir)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path.resolve()

        python_tag = (
            sys.implementation.cache_tag
            or f"py{sys.version_info.major}{sys.version_info.minor}"
        )
        platform_tag = sysconfig.get_platform()

        fingerprint_source = "\0".join(
            [
                os.environ.get("CC", ""),
                os.environ.get("CXX", ""),
                generator or "",
                os.environ.get("CMAKE_TOOLCHAIN_FILE", ""),
                *extra_cmake_args,
            ]
        )
        fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[
            :12
        ]

        return (
            PROJECT_ROOT
            / "build"
            / "cmake"
            / python_tag
            / (f"{platform_tag}-{build_type.lower()}-{fingerprint}")
        )


class EditableWheel(editable_wheel):
    """skip build_ext when installing editable wheel."""

    def _run_build_subcommands(self) -> None:
        build = self.get_finalized_command("build")
        for name in build.get_sub_commands():
            if name == "build_ext":
                continue
            self.run_command(name)


setup(
    use_scm_version={
        "write_to": "catlass/_version.py",
        "fallback_version": "0.0.0",
    },
    ext_modules=[
        CMakeExtension(
            "catlass._tla_type_bridge_native",
            sourcedir="csrc/mlir",
            target="tla-compiler",
            cmake_output_dir="python/catlass",
        ),
    ],
    cmdclass={
        "build_ext": CMakeBuild,
        "editable_wheel": EditableWheel,
    },
    zip_safe=False,
)
