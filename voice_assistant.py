import pyaudio
import dashscope
from dashscope.audio.asr import *
from dashscope.audio.tts_v2 import *
from http import HTTPStatus
import json
from tools import *
from Snowboy import snowboydecoder


class Agent:
    asr_model = "gummy-chat-v1"  # asr模型名称
    tts_model = "cosyvoice-v1"  # tts模型名称
    tts_voice = "longxiaochun"  # tts语音名称
    assistant_model = "qwen-turbo-2025-04-28"  # 大模型名称

    assistant = None  # 大模型助手
    thread = None  # 线程
    tools = None  # 工具列表

    class TTSCallback(ResultCallback):
        _player = None
        _stream = None

        def on_open(self):
            self._player = pyaudio.PyAudio()
            self._stream = self._player.open(
                format=pyaudio.paInt16, channels=1, rate=22050, output=True
            )

        def on_close(self):
            # stop player
            self._stream.stop_stream()
            self._stream.close()
            self._player.terminate()

        def on_data(self, data: bytes) -> None:
            self._stream.write(data)  # 往音频设备中写入数据,即播放音频

    class ASRCallback(TranslationRecognizerCallback):
        _mic = None
        _stream = None
        asr_result = None

        def on_open(self) -> None:
            self._mic = pyaudio.PyAudio()
            self._stream = self._mic.open(
                format=pyaudio.paInt16, channels=1, rate=16000, input=True
            )

        def on_close(self) -> None:
            self._stream.stop_stream()
            self._stream.close()
            self._mic.terminate()
            self._stream = None
            self._mic = None

        def on_event(
            self,
            request_id,
            transcription_result: TranscriptionResult,
            translation_result: TranslationResult,
            usage,
        ) -> None:
            if transcription_result is not None:
                if transcription_result.is_sentence_end:
                    self.asr_result = transcription_result.text

    def __init__(self, instructions: str, tools_json_path: str):
        # 从json中加载预设tools
        with open(tools_json_path, "r", encoding="utf-8") as tools_file:
            self.tools = json.load(tools_file)

        # 创建助手
        self.assistant = dashscope.Assistants.create(
            model=self.assistant_model,  # 大语言模型名，用于为智能体配置大模型
            name='"瞳灵"台灯智能体',  # 智能体名称，用于区别智能体的名称
            description='"瞳灵"台灯智能体,用于进行语音交互',  # 智能体的功能描述
            instructions=instructions,  # 由自然语言构成的指令，用于定义智能体的角色和任务
            tools=self.tools,  # 为智能体配置的工具列表
        )
        # 检测助手创建是否成功
        if not self.__check_status(self.assistant, "助手创建"):
            exit()
        # 创建助手的线程
        self.thread = dashscope.Threads.create()
        # 检测线程创建是否成功
        if not self.__check_status(self.thread, "线程创建"):
            exit()

    def __check_status(self, component, operation):
        if component.status_code == HTTPStatus.OK:
            print(f"{operation} 成功。")
            return True
        else:
            print(
                f"{operation} 失败。状态码：{component.status_code}，错误码：{component.code}，错误信息：{component.message}"
            )
            return False

    def get_message(self):
        # 创建语音识别回调函数
        asr_callback = self.ASRCallback()
        # 创建语音识别模型对象
        translator = TranslationRecognizerChat(
            model="gummy-chat-v1",
            format="pcm",
            sample_rate=16000,
            transcription_enabled=True,
            callback=asr_callback,
            max_end_silence=700,  # 设置最大结束静音时长,单位为毫秒(ms),若语音结束后静音时长超过该预设值,系统将判定当前语句已结束
        )
        translator.start()
        message = ""
        while True:
            if asr_callback._stream:
                data = asr_callback._stream.read(3200, exception_on_overflow=False)
                if not translator.send_audio_frame(data):
                    message = asr_callback.asr_result
                    break
            else:
                break
        translator.stop()
        return message

    def send_message(
        self,
        messages: str,
    ):
        if messages == "":
            return
        # 向线程发送消息
        message = dashscope.Messages.create(self.thread.id, content=messages)
        # 检测消息创建是否成功
        if not self.__check_status(message, "消息创建"):
            exit()

        # 向助手发送消息
        run = dashscope.Runs.create(
            self.thread.id, assistant_id=self.assistant.id, stream=True  # 开启流式输出
        )

        # 创建tts合成器
        tts_callback = self.TTSCallback()
        synthesizer = SpeechSynthesizer(
            model=self.tts_model,
            voice=self.tts_voice,
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
            callback=tts_callback,
        )

        while True:  # 添加外层循环
            for (
                event,
                data,
            ) in run:  # 事件流和事件数据的详细信息
                if (
                    event == "thread.run.requires_action"
                ):  # Assistant 调用了工具，正在等待函数输出
                    tool_outputs = []  # 提交输出的方法与第五步类似
                    for tool in data.required_action.submit_tool_outputs.tool_calls:
                        name = tool.function.name
                        args = json.loads(tool.function.arguments)
                        output = tools_map[name](**args)
                        tool_outputs.append(
                            {
                                "tool_call_id": tool.id,
                                "output": output,
                            }
                        )
                    run = dashscope.Runs.submit_tool_outputs(  # 提交函数输出
                        thread_id=self.thread.id,
                        run_id=data.id,
                        tool_outputs=tool_outputs,
                        stream=True,  # 此处也需要开启流式输出
                    )
                    break  # 跳出当前的 for 循环，下一次循环将轮询新的 Runs 对象。
                elif event == "thread.message.delta":
                    print(data.delta.content.text.value, end="", flush=True)
                    synthesizer.streaming_call(data.delta.content.text.value)
            else:
                break  # 如果首轮 for 循环正常结束（没有触发函数调用），就直接退出 while 循环

        synthesizer.streaming_complete()
        messages = dashscope.Messages.list(self.thread.id)
        if self.__check_status(messages, "消息检索"):
            if messages.data:
                # 显示最后一条消息的内容（助手的响应）
                last_message = messages.data[0]
                return last_message["content"][0]["text"]["value"]

    def reset(self):
        # 创建助手的线程
        self.thread = dashscope.Threads.create()
        # 检测线程创建是否成功
        if not self.__check_status(self.thread, "线程创建"):
            exit()


def main():
    with open("config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    instructions = config["instructions"]
    dashscope.api_key = config["api_key"]

    my_agent = Agent(
        instructions=instructions,
        tools_json_path="tools.json",
    )

    # 关键词模型路径
    kws_models = [
        "Snowboy/resources/models/snowboy.umdl",
        "Snowboy/resources/models/computer.umdl",
    ]

    while True:
        # detector = snowboydecoder.HotwordDetector(kws_models, sensitivity=0.9)
        # print("正在监听唤醒词... 按 Ctrl+C 退出")
        # detector.start(sleep_time=0.03, stop_on_detect=True)
        # detector.terminate()
        # print("唤醒词被检测到，开始语音识别...")
        setence = my_agent.get_message()
        print(f"==>识别结果：{setence}<==")
        my_agent.send_message(setence)


if __name__ == "__main__":
    main()
