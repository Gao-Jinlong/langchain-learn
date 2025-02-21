import { ChatOpenAI } from "@langchain/openai";
import { SerpAPI } from "@langchain/community/tools/serpapi";
import { AgentExecutor } from "langchain/agents";
import { pull } from "langchain/hub";
import { createOpenAIToolsAgent } from "langchain/agents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Calculator } from "@langchain/community/tools/calculator";
import dotenv from "dotenv";

dotenv.config({
  path: "../.env.local",
});
const chatOptions = {
  openAIApiKey: process.env.Tongyi_API_KEY,
  temperature: 1.5,
  modelName: "qwen-plus",
  configuration: {
    baseURL: process.env.BASE_URL,
  },
};

async function main() {
  const searchTool = new SerpAPI(process.env.SERP_KEY);
  searchTool.name = "webSearch";
  const tools = [searchTool, new Calculator()];

  const prompt = await pull<ChatPromptTemplate>("hwchase17/openai-tools-agent");

  const llm = new ChatOpenAI({
    ...chatOptions,
    temperature: 0,
  });

  const agent = await createOpenAIToolsAgent({
    llm,
    tools,
    prompt,
  });

  const agentExecutor = new AgentExecutor({
    agent,
    tools,
  });

  const result = await agentExecutor.invoke({
    input: "我有 10000 人民币，可以购买多少微软股票",
  });

  console.log(result);
}

main();
