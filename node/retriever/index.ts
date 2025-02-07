import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import { LLMChainExtractor } from "langchain/retrievers/document_compressors/chain_extract";
import { ContextualCompressionRetriever } from "langchain/retrievers/contextual_compression";
import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";

/**
 * 通过大模型对问题可能产生的歧义进行扩充
 * 将原有的问题改写为三个更准确的问题
 * 根据改写后的三个问题去查找向量数据库，检索出最相关的文档并去重
 */
async function main() {
  const env = await dotenv.config({
    path: "../.env.local",
  });
  console.log("🚀 ~ main ~ env:", env);

  const directory = "../db/kongyiji";
  const embeddings = new AlibabaTongyiEmbeddings({
    apiKey: process.env.Tongyi_API_KEY,
    modelName: "text-embedding-v2",
  });
  const vectorStore = await FaissStore.load(directory, embeddings);

  const chatOptions = {
    openAIApiKey: process.env.Tongyi_API_KEY,
    model: "deepseek-v3",
    configuration: {
      baseURL: process.env.BASE_URL,
    },
  };

  const model = new ChatOpenAI(chatOptions);
  const retriever = MultiQueryRetriever.fromLLM({
    llm: model,
    retriever: vectorStore.asRetriever(3),
    queryCount: 3,
    verbose: true,
  });

  const res = await retriever.invoke("茴香豆是做什么用的");
  console.log("🚀 ~ main ~ res:", res);
}

/**
 * 通过大模型对问题搜索到的向量数据进行压缩提取，从而减小对话上下文
 */
async function main2() {
  const env = await dotenv.config({
    path: "../.env.local",
  });
  console.log("🚀 ~ main ~ env:", env);

  const directory = "../db/kongyiji";
  const embeddings = new AlibabaTongyiEmbeddings({
    apiKey: process.env.Tongyi_API_KEY,
    modelName: "text-embedding-v2",
  });
  const vectorStore = await FaissStore.load(directory, embeddings);

  const chatOptions = {
    openAIApiKey: process.env.Tongyi_API_KEY,
    model: "deepseek-v3",
    configuration: {
      baseURL: process.env.BASE_URL,
    },
  };

  const model = new ChatOpenAI(chatOptions);

  const compressor = LLMChainExtractor.fromLLM(model);
  const retriever = new ContextualCompressionRetriever({
    baseCompressor: compressor,
    baseRetriever: vectorStore.asRetriever(2),
  });

  const res = await retriever.invoke("茴香豆是做什么用的");
  console.log("🚀 ~ main ~ res:", res);
}

/**
 * 动态调整召回的文档数量，例如原文茴香豆的内容可能只有两只三个，但孔乙己相关的内容却有几十条，因此动态调整召回的文档数量可以获得更好的检索效果
 */
async function main3() {
  const env = await dotenv.config({
    path: "../.env.local",
  });
  console.log("🚀 ~ main ~ env:", env);

  const directory = "../db/kongyiji";
  const embeddings = new AlibabaTongyiEmbeddings({
    apiKey: process.env.Tongyi_API_KEY,
    modelName: "text-embedding-v2",
  });
  const vectorStore = await FaissStore.load(directory, embeddings);

  const chatOptions = {
    openAIApiKey: process.env.Tongyi_API_KEY,
    model: "deepseek-v3",
    configuration: {
      baseURL: process.env.BASE_URL,
    },
  };

  const model = new ChatOpenAI(chatOptions);

  const retriever = ScoreThresholdRetriever.fromVectorStore(vectorStore, {
    minSimilarityScore: 0.4,
    maxK: 5,
    kIncrement: 1,
  });

  const res = await retriever.invoke("茴香豆是做什么用的");
  console.log("🚀 ~ main ~ res:", res);
}

main3();
