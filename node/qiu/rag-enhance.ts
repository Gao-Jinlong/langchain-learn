import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import path from "path";
import dotenv from "dotenv";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { Document } from "@langchain/core/documents";
import {
  RunnablePassthrough,
  RunnableSequence,
  RunnableWithMessageHistory,
} from "@langchain/core/runnables";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { JSONChatHistory } from "../utils/JSONChatHistory";

dotenv.config({
  path: "../.env.local",
});

const chatHistoryDir = path.join(__dirname, "../../data/chat_history");
const chatOptions = {
  openAIApiKey: process.env.Tongyi_API_KEY,
  temperature: 1.5,
  modelName: "deepseek-v3",
  configuration: {
    baseURL: process.env.BASE_URL,
  },
};

const SYSTEM_TEMPLATE = `
你是一个熟读刘慈欣的《球状闪电》的终极原着党，精通根据作品原文详细解释和回答问题，你在回答时会引用作品原文。
并且回答时仅根据原文，尽可能回答用户问题，如果原文中没有相关内容，你可以回答“原文中没有相关内容”，

以下是原文中跟用户回答相关的内容：
{context}
`;
const prompt = ChatPromptTemplate.fromMessages([
  ["system", SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  ["human", "现在，你需要基于原文，回答以下问题：\n{standalone_question}`"],
]);

async function loadVectorStore() {
  const directory = path.join(__dirname, "../../db/qiu");
  const embedding = new AlibabaTongyiEmbeddings({
    apiKey: process.env.Tongyi_API_KEY,
    modelName: "text-embedding-v2",
  });

  const vectorStore = await FaissStore.load(directory, embedding);

  return vectorStore;
}

const rephraseChainPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "给定以下对话和一个后续问题，请将后续问题重述为一个独立的问题。请注意，重述的问题应该包含足够的信息，使得没有看过对话历史的人也能理解。",
  ],
  new MessagesPlaceholder("history"),
  ["human", "将以下问题重述为一个独立的问题：\n{question}"],
]);
const rephraseChain = RunnableSequence.from([
  rephraseChainPrompt,
  new ChatOpenAI({
    ...chatOptions,
    temperature: 0.2,
  }),
  new StringOutputParser(),
]);

async function main() {
  const vectorStore = await loadVectorStore();
  const retriever = vectorStore.asRetriever(2);

  const convertDocsToString = (documents: Document[]) => {
    return documents.map((doc) => doc.pageContent).join("\n");
  };

  // 处理用户的输入问题，将问题转换为更利于检索的文本
  const contextRetrieverChain = RunnableSequence.from([
    (input) => input.standalone_question,
    retriever,
    convertDocsToString,
  ]);

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    ["human", "现在，你需要基于原文，回答一下问题：\n{standalone_question}"],
  ]);

  const model = new ChatOpenAI(chatOptions);

  const ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: rephraseChain,
    }),
    RunnablePassthrough.assign({
      context: contextRetrieverChain,
    }),
    prompt,
    model,
    new StringOutputParser(),
  ]);

  const ragChainWithHistory = new RunnableWithMessageHistory({
    runnable: ragChain,
    getMessageHistory: (sessionId) =>
      new JSONChatHistory({ sessionId, dir: chatHistoryDir }),
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });

  const res = await ragChainWithHistory.invoke(
    {
      question: "什么是球状闪电？",
    },
    {
      configurable: { sessionId: "test-history" },
    }
  );
}

main();
