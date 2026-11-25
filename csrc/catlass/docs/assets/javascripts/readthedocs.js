document.addEventListener("DOMContentLoaded", () => {
  const searchInput = document.querySelector(".md-search__input");
  if (!searchInput) {
    return;
  }

  searchInput.addEventListener("focus", () => {
    document.dispatchEvent(new CustomEvent("readthedocs-search-show"));
  });
});
