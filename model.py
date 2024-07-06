"""
定义一个model类，用来离线调用自定义的本地大模型
"""
from typing import Optional, List, Any

from langchain_community.llms.utils import enforce_stop_tokens
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from transformers import AutoTokenizer, AutoModel


class ChatGLM3(LLM):
    # 定于全局变量
    tokenizer: object = None
    model: object = None
    history: list = []

    # 构造函数
    def __init__(self):
        super().__init__()

    # 重构_llm_type函数
    @property
    def _llm_type(self):
        return "custom chatglm3"

    # 定义load_model函数
    def load_model(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision='main')
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).quantize(4).cuda()
        self.model.eval()

    # 重构_call函数
    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            model_path=r"./chatglm3-6b",
            **kwargs: Any,
    ) -> str:
        self.load_model(model_path=model_path)
        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            query=prompt,
        )
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history.append([(None, response)])

        return response


# # 测试模型调用
# if __name__ == '__main__':
#     # 实例化大语言模型对象
#     llm = ChatGLM3()
#     # 调用大语言模型对象的_call方法
#     query = "你好，这条消息是用来测试模型调用是否成功。没有问题请回复：成功。"
#     response = llm.invoke(query)
#     print(f">>>>> Answer: {response}")
