import { ChatAlibabaTongyi } from "@langchain/community/chat_models/alibaba_tongyi";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { readFileSync } from "fs";
import path from "path";
import dotenv from "dotenv";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import readline from "readline";
import util from "util";

dotenv.config({
  path: "../.env.local",
});

process.env.LANGCHAIN_TRACING_V2 = "true";

const tongyiChatOptions = {
  alibabaApiKey: process.env.Tongyi_API_KEY,
  temperature: 1.5,
  modelName: "qwen-plus",
};

const guaInfoBuffer = readFileSync(path.join(__dirname, "./gua.json"));
const guaInfo = JSON.parse(guaInfoBuffer.toString());

const yaoName = ["初爻", "二爻", "三爻", "四爻", "五爻", "六爻"];

const guaDict = {
  阳阳阳: "乾",
  阴阴阴: "坤",
  阴阳阳: "兑",
  阳阴阳: "震",
  阳阳阴: "巽",
  阴阳阴: "坎",
  阳阴阴: "艮",
  阴阴阳: "离",
};

function generateGua(): string[] {
  let yaoCount = 0;
  const messageList = [];

  const genYao = () => {
    const coinRes = Array.from({ length: 3 }, () =>
      Math.random() > 0.5 ? 1 : 0
    );
    const yinYang = coinRes.reduce((a, b) => a + b, 0) > 1.5 ? "阳" : "阴";
    const message = `${yaoName[yaoCount]} 为 ${coinRes
      .map((i) => (i > 0.5 ? "字" : "背"))
      .join("")} 为 ${yinYang}`;

    return {
      yinYang,
      message,
    };
  };

  const firstGuaYinYang = Array.from({ length: 3 }, () => {
    const { yinYang, message } = genYao();
    yaoCount++;

    messageList.push(message);
    return yinYang;
  });
  const firstGua = guaDict[firstGuaYinYang.join("")];
  messageList.push(`您的首卦为 ${firstGua}`);

  const secondGuaYinYang = Array.from({ length: 3 }, () => {
    const { yinYang, message } = genYao();
    yaoCount++;

    messageList.push(message);
    return yinYang;
  });
  const secondGua = guaDict[secondGuaYinYang.join("")];
  messageList.push(`您的次卦为 ${secondGua}`);

  const gua = secondGua + firstGua;
  const guaDesc = guaInfo[gua];

  const guaRes = `
六爻结果: ${gua}  
卦名为：${guaDesc.name}   
${guaDesc.des}   
卦辞为：${guaDesc.sentence}   
  `;

  messageList.push(guaRes);

  return messageList;
}

async function main() {
  const messageList = generateGua();
  const history = new ChatMessageHistory();

  const guaMessage = messageList.map((message): ["ai", string] => [
    "ai",
    message,
  ]);

  const prompt = await ChatPromptTemplate.fromMessages([
    [
      "system",
      `你是一位出自中华六爻世家的卜卦专家，你的任务是根据卜卦者的问题和得到的卦象，为他们提供有益的建议。
你的解答应基于卦象的理解，同时也要尽可能地展现出乐观和积极的态度，引导卜卦者朝着积极的方向发展。
你的语言应该具有仙风道骨、雅致高贵的气质，以此来展现你的卜卦专家身份。`,
    ],
    ...guaMessage,
    new MessagesPlaceholder("history_message"),
    ["human", "{input}"],
  ]);

  const llm = new ChatAlibabaTongyi(tongyiChatOptions);
  const chian = prompt.pipe(llm).pipe(new StringOutputParser());
  const chainWithHistory = new RunnableWithMessageHistory({
    runnable: chian,
    getMessageHistory: (_sessionId) => history,
    inputMessagesKey: "input",
    historyMessagesKey: "history_message",
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = util.promisify(rl.question).bind(rl);

  const input = await question("告诉我你的疑问：");

  let index = 0;
  const printMessagePromise = new Promise<void>((resolve) => {
    const intervalId = setInterval(() => {
      if (index < messageList.length) {
        console.log(messageList[index]);
        index++;
      } else {
        clearInterval(intervalId);
        resolve();
      }
    }, 1000);
  });

  const llmResPromise = chainWithHistory.invoke(
    {
      input: "用户的问题是：" + input,
    },
    {
      configurable: { sessionId: "no-used" },
    }
  );

  const [_, firstRes] = await Promise.all([printMessagePromise, llmResPromise]);

  console.log(firstRes);

  async function chat() {
    const input = await question("User: ");

    if (input.toLowerCase() === "exit") {
      rl.close();
      return;
    }

    const response = await chainWithHistory.invoke(
      { input },
      { configurable: { sessionId: "no-used" } }
    );

    console.log("AI: ", response);
    chat();
  }

  chat();
}

main();
