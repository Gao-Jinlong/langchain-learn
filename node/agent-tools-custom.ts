import { DynamicTool } from "@langchain/core/tools";
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import path from "path";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import dotenv from "dotenv";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

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

async function main() {
  const prompt =
    ChatPromptTemplate.fromTemplate(`将以下问题仅基于提供的上下文进行回答：
    上下文：
    {context}

    问题：{input}`);
  const llm = new ChatOpenAI(chatOptions);

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
}

main();
