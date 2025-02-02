import { TextLoader } from "langchain/document_loaders/fs/text"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import FaissStore from "faiss-node"



const run = async ()=>{
  const loader = new TextLoader("../data/kong.txt")
  const docs = await loader.load()

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  })

  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new AlibabaTongyiEmbeddings({
    apiKey: process.env.Tongyi_API_KEY,
    modelName: "text-embedding-v2",
  })
  const vectorStore = await FaissStore.fromDocuments()
}