import { BaseListChatMessageHistory } from "@langchain/core/chat_history";
import {
  BaseMessage,
  mapChatMessagesToStoredMessages,
  mapStoredMessagesToChatMessages,
  StoredMessage,
} from "@langchain/core/messages";
import path from "path";
import fs from "fs";

export interface JSONChatHistoryInput {
  sessionId: string;
  dir: string;
}

export class JSONChatHistory extends BaseListChatMessageHistory {
  lc_namespace = ["langchain", "stores", "message"];

  sessionId: string;
  dir: string;

  constructor(fields: JSONChatHistoryInput) {
    super(fields);
    this.sessionId = fields.sessionId;
    this.dir = fields.dir;
  }

  async getMessages(): Promise<BaseMessage[]> {
    const filePath = path.join(this.dir, `${this.sessionId}.json`);
    try {
      if (!fs.existsSync(filePath)) {
        this.saveMessagesToFile([]);
        return [];
      }

      const data = fs.readFileSync(filePath, { encoding: "utf-8" });
      const storedMessages = JSON.parse(data) as StoredMessage[];
      return mapStoredMessagesToChatMessages(storedMessages);
    } catch (e) {
      console.log(`Failed to read chat history from ${filePath}:`, e);
      return [];
    }
  }

  async addMessage(message: BaseMessage): Promise<void> {
    const messages = await this.getMessages();
    messages.push(message);
    await this.saveMessagesToFile(messages);
  }

  async addMessages(messages: BaseMessage[]): Promise<void> {
    const existingMessages = await this.getMessages();
    const allMessages = existingMessages.concat(messages);
    await this.saveMessagesToFile(allMessages);
  }

  async clear(): Promise<void> {
    const filePath = path.join(this.dir, `${this.sessionId}.json`);
    try {
      fs.unlinkSync(filePath);
    } catch (error) {
      console.error(`Failed to clear chat history from ${filePath}`, error);
    }
  }

  private async saveMessagesToFile(messages: BaseMessage[]): Promise<void> {
    const filePath = path.join(this.dir, `${this.sessionId}.json`);
    const serializedMessages = mapChatMessagesToStoredMessages(messages);
    try {
      fs.writeFileSync(filePath, JSON.stringify(serializedMessages, null, 2), {
        encoding: "utf-8",
      });
    } catch (error) {
      console.error(`Failed to save chat history to ${filePath}`, error);
    }
  }
}
