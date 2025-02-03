import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import dotenv from "dotenv";

async function main() {
  const env = await dotenv.config({
    path: "../.env.local",
  });

  console.log("env", env);

  const directory = "../db/kongyiji";
  const embedding = new AlibabaTongyiEmbeddings({
    apiKey: process.env.Tongyi_API_KEY,
  });

  const vectorStore = await FaissStore.load(directory, embedding);

  const retriever = vectorStore.asRetriever(2);

  const result = await retriever.invoke("茴香豆是做什么用的");

  console.log(result);
}

main();
