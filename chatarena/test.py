from chatarena.backends.langchain_ollama import LangChainOpenAIChat
from chatarena.message import Message

xx = LangChainOpenAIChat()

mm = Message("Player", "what is leadership", turn = 1)

xx.query("player", "you are a teacher.", [mm])