import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import dotenv from "dotenv";
import { DocumentInterface } from "@langchain/core/documents";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

async function main() {
  dotenv.config({
    path: "../.env.local",
  });

  const loader = new TextLoader("../data/qiu.txt");
  const docs = await loader.load();

  const spiltter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });

  const splitDocs = await spiltter.splitDocuments(docs);

  const embeddings = new AlibabaTongyiEmbeddings({
    apiKey: process.env.Tongyi_API_KEY,
    modelName: "text-embedding-v2",
  });

  const vectorStore = new MemoryVectorStore(embeddings);
  await vectorStore.addDocuments(splitDocs);

  const retriever = vectorStore.asRetriever(2);

  const convertDocsToString = (documents: DocumentInterface[]) => {
    return documents.map((document) => document.pageContent).join("\n");
  };

  const contextRetriverChain = RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString,
  ]);

  const TEMPLATE = `
  你是一个熟读刘慈欣的《球状闪电》的终极原著党，精通根据作品原文详细解释和回答问题，你在回答时会引用作品原文。
  并且回答时仅根据原文，尽可能回答用户问题，如果原文中没有相关内容，你可以回答“原文中没有相关内容”，
  
  以下是原文中跟用户回答相关的内容：
  {context}
  
  现在，你需要基于原文，回答以下问题：
  {question}`;

  const prompt = ChatPromptTemplate.fromTemplate(TEMPLATE);

  const chatOptions = {
    openAIApiKey: process.env.Tongyi_API_KEY,
    model: "deepseek-v3",
    configuration: {
      baseURL: process.env.BASE_URL,
    },
  };

  const model = new ChatOpenAI(chatOptions);

  const ragChain = RunnableSequence.from([
    {
      context: contextRetriverChain,
      question: (input) => input.question,
    },
    prompt,
    model,
  ]);

  const res = await ragChain.invoke({
    question: "什么是球状闪电",
  });

  console.log("res", res);
}

main();
