import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { SqlToolkit } from "langchain/agents/toolkits/sql";
import { pull } from "langchain/hub";
import { SqlDatabase } from "langchain/sql_db";
import { QuerySqlTool } from "langchain/tools/sql";
import { DataSource } from "typeorm";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

function get_llm(): ChatAnthropic {
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY environment variable is not set");
  }

  const llm = new ChatAnthropic({
    model: "claude-3-5-sonnet-20240620",
    temperature: 0,
  });

  return llm;
}

async function get_db(): Promise<SqlDatabase> {
  const datasource = new DataSource({
    type: "sqlite",
    database: "./Chinook.db",
  });
  const db = await SqlDatabase.fromDataSourceParams({
    appDataSource: datasource,
  });

  return db;
}

interface QueryResult {
  query: string;
  result: Record<string, any>[];
}

async function generate_sql_query(question: string, db: SqlDatabase, llm: ChatAnthropic): Promise<QueryResult> {
  const queryPromptTemplate = await pull<ChatPromptTemplate>(
    "langchain-ai/sql-query-system-prompt",
  );

  const queryOutput = z.object({
    query: z.string().describe("Syntactically valid SQL query."),
  });

  const structuredLlm = llm.withStructuredOutput(queryOutput);

  const promptValue = await queryPromptTemplate.invoke({
    dialect: db.appDataSourceOptions.type,
    top_k: 10,
    table_info: await db.getTableInfo(),
    input: question,
  });

  const result = await structuredLlm.invoke(promptValue);
  const executeQueryTool = new QuerySqlTool(db);
  const queryResult = await executeQueryTool.invoke(result.query);
  
  const parsedResult = typeof queryResult === 'string' ? JSON.parse(queryResult) : queryResult;
  
  return {
    query: result.query,
    result: Array.isArray(parsedResult) ? parsedResult : [parsedResult]
  };
}

interface AgentResult {
  queries: QueryResult[];
  finalAnswer: string;
}

async function agent_results(question: string): Promise<AgentResult> {
  const llm = get_llm();
  const db = await get_db();
  const toolkit = new SqlToolkit(db, llm);
  const tools = toolkit.getTools();

  const systemPromptTemplate = await pull<ChatPromptTemplate>(
    "langchain-ai/sql-agent-system-prompt",
  );

  const systemMessage = await systemPromptTemplate.format({
    dialect: "SQLite",
    top_k: 5,
  });

  const agent = createReactAgent({
    llm: llm,
    tools: tools,
    stateModifier: systemMessage,
  });

  const inputs = {
    messages: [
      {
        role: "user",
        content: question,
      },
    ],
  };

  const queries: QueryResult[] = [];
  let agentResponse = "";
  let currentQuery = "";

  for await (const step of await agent.stream(inputs, {
    streamMode: "values",
  })) {
    const lastMessage = step.messages[step.messages.length - 1];

    if (
      lastMessage._getType() === "ai" &&
      (lastMessage as AIMessage).tool_calls?.length
    ) {
      const toolCalls = (lastMessage as AIMessage).tool_calls || [];
      for (const tc of toolCalls) {
        if (
          (tc.name === "query-sql" || tc.name === "query-checker") &&
          tc.args?.input
        ) {
          currentQuery = tc.args.input;
        }
      }
    }

    if (lastMessage._getType() === "tool") {
      const toolMessage = lastMessage as any;
      if (toolMessage.name === "query-sql" && currentQuery) {
        try {
          const result = JSON.parse(toolMessage.content);
          queries.push({
            query: currentQuery,
            result: result,
          });
        } catch {
          queries.push({
            query: currentQuery,
            result: [{ result: toolMessage.content }],
          });
        }
        currentQuery = "";
      }
    }

    if (
      lastMessage._getType() === "ai" &&
      !((lastMessage as AIMessage).tool_calls?.length || 0 > 0)
    ) {
      agentResponse = lastMessage.content as string;
    }
  }

  return {
    queries,
    finalAnswer: agentResponse,
  };
}

async function copyToClipboard(text: string): Promise<void> {
  try {
    if (process.platform === 'darwin') {
      await execAsync(`echo '${text.replace(/'/g, "'\\''")}' | pbcopy`);
    } else if (process.platform === 'linux') {
      await execAsync(`echo '${text.replace(/'/g, "'\\''")}' | xclip -selection clipboard`);
    } else if (process.platform === 'win32') {
      await execAsync(`echo '${text.replace(/'/g, "'\\''")}' | clip`);
    }
  } catch (error) {
    console.warn("Could not copy to clipboard:", error);
  }
}

async function askQuestion(question: string) {
  console.log(`\n🤔 Question: ${question}\n`);
  
  try {
    // Get basic SQL query
    const db = await get_db();
    const llm = get_llm();
    const basicResult = await generate_sql_query(question, db, llm);
    
    console.log("📝 Generated SQL Query:");
    console.log("```sql");
    console.log(basicResult.query);
    console.log("```\n");
    
    // Copy SQL to clipboard
    await copyToClipboard(basicResult.query);
    console.log("📋 SQL query copied to clipboard\n");
    
    console.log("📊 Query Results:");
    console.log(JSON.stringify(basicResult.result, null, 2));
    console.log("\n");
    
    // Get agent response
    console.log("🤖 Getting agent response...\n");
    const agentResult = await agent_results(question);
    
    if (agentResult.queries.length > 0) {
      console.log("🔍 Agent SQL Queries:");
      agentResult.queries.forEach((query, index) => {
        console.log(`Query ${index + 1}: ${query.query}`);
      });
      console.log("\n");
    }
    
    console.log("💬 Agent Response:");
    console.log(agentResult.finalAnswer);
    console.log("\n" + "=".repeat(80));
    
  } catch (error) {
    console.error("❌ Error:", error);
  }
}

async function main() {
  const question = process.argv[2];
  
  if (!question) {
    console.log("Usage: npx tsx ask.ts \"Your question here\"");
    console.log("Example: npx tsx ask.ts \"How many employees are there?\"");
    process.exit(1);
  }
  
  await askQuestion(question);
}

main().catch(console.error);