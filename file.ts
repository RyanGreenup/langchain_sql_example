import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { SqlToolkit } from "langchain/agents/toolkits/sql";
import { pull } from "langchain/hub";
import { SqlDatabase } from "langchain/sql_db";
import { QuerySqlTool } from "langchain/tools/sql";
import { DataSource } from "typeorm";
import { z } from "zod";

import { createReactAgent } from "@langchain/langgraph/prebuilt";

const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  query: Annotation<string>,
  result: Annotation<string>,
  answer: Annotation<string>,
});

function get_llm(): ChatAnthropic {
  // Depends on ANTHROPIC_API_KEY env var
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY environment variable is not set");
  }

  // Connect to the LLM
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

async function main() {
  // Query the database
  const db = await get_db();
  const result = await db.run("SELECT * FROM Artist LIMIT 10;");
  console.log(result);

  // Look at the template
  const queryPromptTemplate = await pull<ChatPromptTemplate>(
    "langchain-ai/sql-query-system-prompt",
  );

  queryPromptTemplate.promptMessages.forEach((message) => {
    console.log(message.lc_kwargs.prompt.template);
  });

  // Connect to the LLM
  const llm = get_llm();

  // Query the LLM

  // Step 1: Define the expected output structure using Zod schema
  // This ensures the LLM returns a properly formatted SQL query
  const queryOutput = z.object({
    query: z.string().describe("Syntactically valid SQL query."),
  });

  // Step 2: Create a structured LLM that will enforce the output schema
  // This wraps our LLM to guarantee structured responses
  const structuredLlm = llm.withStructuredOutput(queryOutput);

  // Step 3: Define the query writing function
  // This function takes a question and converts it to a SQL query
  const writeQuery = async (state: typeof InputStateAnnotation.State) => {
    // Step 3a: Prepare the prompt with database context and user question
    const promptValue = await queryPromptTemplate.invoke({
      dialect: db.appDataSourceOptions.type, // Database type (sqlite)
      top_k: 10, // Limit results to 10 rows
      table_info: await db.getTableInfo(), // Database schema information
      input: state.question, // User's natural language question
    });

    // Step 3b: Send the prompt to the LLM and get structured response
    const result = await structuredLlm.invoke(promptValue);

    // Step 3c: Return the generated SQL query
    return { query: result.query };
  };

  // Step 4: Test the query generation with an example question
  const result_2 = await writeQuery({
    question: "How many Employees are there?",
  });
  console.log(result_2);

  // Execute the query

  const executeQuery = async (state: typeof StateAnnotation.State) => {
    const executeQueryTool = new QuerySqlTool(db);
    return { result: await executeQueryTool.invoke(state.query) };
  };

  const result_3 = await executeQuery({
    question: "",
    query: "SELECT COUNT(*) AS EmployeeCount FROM Employee;",
    result: "",
    answer: "",
  });

  console.log(result_3);

  // Answer the Question
  const generateAnswer = async (state: typeof StateAnnotation.State) => {
    const promptValue =
      "Given the following user question, corresponding SQL query, " +
      "and SQL result, answer the user question.\n\n" +
      `Question: ${state.question}\n` +
      `SQL Query: ${state.query}\n` +
      `SQL Result: ${state.result}\n`;
    const response = await llm.invoke(promptValue);
    return { answer: response.content };
  };

  // Langchain Graph
  const graphBuilder = new StateGraph({
    stateSchema: StateAnnotation,
  })
    .addNode("writeQuery", writeQuery)
    .addNode("executeQuery", executeQuery)
    .addNode("generateAnswer", generateAnswer)
    .addEdge("__start__", "writeQuery")
    .addEdge("writeQuery", "executeQuery")
    .addEdge("executeQuery", "generateAnswer")
    .addEdge("generateAnswer", "__end__");
  const graph = graphBuilder.compile();

  let inputs = {
    question: "Show how to join two tables",
  };
  console.log(
    "-----------------------------------------------------------------",
  );
  console.log(
    "-----------------------------------------------------------------",
  );
  console.log(
    "-----------------------------------------------------------------",
  );

  console.log(inputs);
  console.log("\n====\n");
  // Note that each step is
  let finalResult: any[] = [];
  for await (const step of await graph.stream(inputs, {
    streamMode: "updates",
  })) {
    console.log(step);
    console.log("\n====\n");
    finalResult.push(step);
  }
  // Print only the output text here
  console.log("Final Answer:");
  console.log(finalResult);
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////// Agent /////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

function formatJsonAsMarkdownTable(jsonString: string): string {
  try {
    const data = JSON.parse(jsonString);

    if (!Array.isArray(data) || data.length === 0) {
      return "No tabular data available";
    }

    const firstRow = data[0];
    if (typeof firstRow !== "object" || firstRow === null) {
      return "Data is not in tabular format";
    }

    const headers = Object.keys(firstRow);

    // Create header row
    const headerRow = "| " + headers.join(" | ") + " |";

    // Create separator row
    const separatorRow = "| " + headers.map(() => "---").join(" | ") + " |";

    // Create data rows
    const dataRows = data.map((row) => {
      const values = headers.map((header) => {
        const value = row[header];
        return value !== null && value !== undefined ? String(value) : "";
      });
      return "| " + values.join(" | ") + " |";
    });

    return [headerRow, separatorRow, ...dataRows].join("\n");
  } catch (error) {
    return "Error formatting data as table: " + error;
  }
}

interface QueryResult {
  query: string;
  result: Record<string, any>[];
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

    // Capture SQL queries from AI messages with tool calls
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

    // Capture SQL results from tool messages
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
          // If not JSON, store as string in an array
          queries.push({
            query: currentQuery,
            result: [{ result: toolMessage.content }],
          });
        }
        currentQuery = "";
      }
    }

    // Capture final AI response
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

async function agent(question: string) {
  // Get results from agent_results function
  const results = await agent_results(question);

  // Print formatted output
  console.log("# 📊 SQL Agent Analysis Summary");

  if (results.queries.length > 0) {
    console.log("\n## 🔍 SQL Queries Executed\n");
    results.queries.forEach((queryResult, index) => {
      console.log(`### Query ${index + 1}:`);
      console.log("```sql");
      console.log(queryResult.query);
      console.log("```\n");
    });

    console.log("## 📋 Query Results\n");
    results.queries.forEach((queryResult, index) => {
      console.log(`### Result ${index + 1}:`);

      // Try to format as table if it's JSON data
      const resultJson = JSON.stringify(queryResult.result);
      const tableFormatted = formatJsonAsMarkdownTable(resultJson);
      if (
        tableFormatted.includes("|") &&
        !tableFormatted.startsWith("Error") &&
        !tableFormatted.startsWith("No tabular") &&
        !tableFormatted.startsWith("Data is not")
      ) {
        console.log(tableFormatted);
        console.log("\n**Raw JSON:**");
        console.log("```json");
        console.log(resultJson);
        console.log("```\n");
      } else {
        console.log("```");
        console.log(resultJson);
        console.log("```\n");
      }
    });
  }

  if (results.finalAnswer) {
    console.log("## 🤖 Agent Response\n");
    console.log(results.finalAnswer);
  }

  console.log("\n" + "=".repeat(80));
}

// async function generate_sql_query_and_output_no_agent()  {
// }

//
// main().catch(console.error);
agent(
  "Which supplier should we charge more money? Assume tasks are paid on a for job basis, infer other details that may drive up costs",
).catch(console.error);
