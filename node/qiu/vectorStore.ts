import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import path from "path";
import dotenv from "dotenv";
import { FaissStore } from "@langchain/community/vectorstores/faiss";

const baseDir = __dirname;

async function main() {
  dotenv.config({
    path: "../.env.local",
  });

  const loader = new TextLoader(path.join(baseDir, "../../data/qiu.txt"));
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  const embedding = new AlibabaTongyiEmbeddings({
    apiKey: process.env.Tongyi_API_KEY,
    modelName: "text-embedding-v2",
  });
  const vectorStore = await FaissStore.fromDocuments(splitDocs, embedding);

  await vectorStore.save(path.join(baseDir, "../../db/qiu"));
}

main();
