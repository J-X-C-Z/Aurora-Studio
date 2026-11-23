# R25 G8 ReadProcessMemory capture

Status: `BLOCKED_NO_CAPTURE_ONE_PROCESS`

## Result

新增 `tools/c1_readprocess_capture.py`。它只申请 `PROCESS_QUERY_LIMITED_INFORMATION |
PROCESS_VM_READ`，枚举 `portrait.exe` 模块基址，远程校验 PE32+、固定 PE SHA、producer
`0x823A3B0` 与 consumer `0x82AA2E0` 的可执行 section，再在限定私有 RW 内存中扫描
shape/stride 一致的 `cv::Mat` header 或常见 draw-buffer 指针结构。ROI 逐行经
`ReadProcessMemory` 双读比对后输出 canonical SHA manifest。

工具不调用 `WriteProcessMemory`、远程线程、调试附加或进程挂起。候选歧义、PE/RVA
不匹配、ROI 不可读、读取不稳定和扫描上限均 fail-closed。

## Verdict

最终验收时没有运行中的 `CaptureOne.exe`/`portrait.exe`，因此没有观察到有效
Mat/draw-buffer 指针，也没有捕获像素。新增可执行 PF 为 **0**；机器状态为
`BLOCKED_NO_CAPTURE_ONE_PROCESS`。

## Unclosed

需要用户打开含目标照片的 Capture One 并触发一次人像分析。即使工具找到稳定 ROI，
它只产生 `CAPTURED_READ_ONLY_UNVERIFIED`；同 job id、output 4/local-Mat 身份、
channel 和 draw warp 仍需独立证据，不能据此增加 PF。

## Verification

仅执行 Python 编译、自测、target doctor、JSON 解析、diff 检查和 apply-check。
未运行 cargo/GPU，未写主树，未修改或注入 Capture One。
