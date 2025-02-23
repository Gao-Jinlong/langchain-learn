import { DynamicStructuredTool, DynamicTool } from "@langchain/core/tools";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import path from "path";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import dotenv from "dotenv";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import z from "zod";
import { Calculator } from "@langchain/community/tools/calculator";
import { pull } from "langchain/hub";
import { AgentExecutor, createOpenAIToolsAgent } from "langchain/agents";
import { ChatAlibabaTongyi } from "@langchain/community/chat_models/alibaba_tongyi";

dotenv.config({
  path: "../.env.local",
});
const chatOptions = {
  openAIApiKey: process.env.Tongyi_API_KEY,
  temperature: 0,
  modelName: "qwen-plus",
  configuration: {
    baseURL: process.env.BASE_URL,
  },
};
// 将 chatOptions 修改为:
const tongyiChatOptions = {
  alibabaApiKey: process.env.Tongyi_API_KEY,
  temperature: 0,
  modelName: "qwen-plus",
};

const stringReverseTool = new DynamicTool({
  name: "string-reverser",
  description:
    "reverses a string. input should be the string you want to reverse.",
  func: async (input: string) => input.split("").reverse().join(""),
});

async function loadVectorStore() {
  const directory = path.join(__dirname, "../db/qiu");
  const embeddings = new AlibabaTongyiEmbeddings({
    apiKey: process.env.Tongyi_API_KEY,
    modelName: "text-embedding-v2",
  });
  const vectorStore = await FaissStore.load(directory, embeddings);

  return vectorStore;
}

async function createToolAgentWithTool(tools) {
  const prompt = await pull<ChatPromptTemplate>("hwchase17/openai-tools-agent");

  const llm = new ChatAlibabaTongyi(tongyiChatOptions);
  const agent = await createOpenAIToolsAgent({
    llm,
    tools,
    prompt,
  });

  const agentExecutor = new AgentExecutor({
    agent,
    tools,
  });
  return agentExecutor;
}

async function main() {
  const prompt =
    ChatPromptTemplate.fromTemplate(`将以下问题仅基于提供的上下文进行回答：
    上下文：
    {context}

    问题：{input}`);
  const llm = new ChatAlibabaTongyi(tongyiChatOptions);

  const documentChain = await createStuffDocumentsChain({
    llm,
    prompt,
  });

  const vectorStore = await loadVectorStore();
  const retriever = vectorStore.asRetriever();

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });

  const retrieverTool = new DynamicTool({
    name: "get-qiu-answer",
    func: async (input: string) => {
      const res = await retrievalChain.invoke({
        input,
      });

      return res;
    },
    description: "获取小说 《球状闪电》相关问题的答案",
  });

  const dateDiffTool = new DynamicStructuredTool({
    name: "date-difference-calculator",
    description: "计算两个日期之间的天数差",
    schema: z.object({
      date1: z.string().describe("第一个日期，以YYYY-MM-DD格式表示"),
      date2: z.string().describe("第二个日期，以YYYY-MM-DD格式表示"),
    }),
    func: async ({ date1, date2 }) => {
      const d1 = new Date(date1);
      const d2 = new Date(date2);
      const difference = Math.abs(d2.getTime() - d1.getTime());
      const days = Math.ceil(difference / (1000 * 60 * 60 * 24));
      return days.toString();
    },
  });

  const tools = [retrieverTool, dateDiffTool, new Calculator()];

  const agents = await createToolAgentWithTool(tools);

  // const res = await agents.invoke({
  //   input: "小说球状闪电中量子玫瑰的情节",
  // });

  // const res = await agents.invoke({
  //   input: "我有 17 个苹果，小明的苹果比我的三倍少 10 个，小明有多少个苹果？",
  // });

  const res = await agents.invoke({
    input: "今年是 2024 年，今年 5.1 和 10.1 之间有多少天？",
  });

  console.log(res);
}

main();
