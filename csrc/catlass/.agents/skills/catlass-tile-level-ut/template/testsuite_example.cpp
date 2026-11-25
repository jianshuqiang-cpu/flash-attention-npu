
TEST_P(${UnittestName}, ${TestSuiteName})
{
    // Previous operations (initialize tensors, set compile-time variables, etc.)
    
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();

    ASSERT_EQ(logs.size(), /* 期望调用次数：1 / row / col / fractal 循环数 */);
    ASSERT_EQ(logs[0].name, /* "DataCopy" / "LoadData" / "Fixpipe" / ... */);
    ASSERT_EQ(logs[0].args.size(), /* API的参数量 */);

    const auto* params = logs[0].GetArgsAt(2).Value</* 对应 Params 类型 */>();
    // ... 逐字段 ASSERT_EQ ...

    ASSERT_EQ(logs[0].GetArgsTAt(0).Type(), typeid(Element));  // Element 校验
}