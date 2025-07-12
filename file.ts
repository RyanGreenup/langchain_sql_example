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


async function main() {
  const db = await get_db();
  const llm = get_llm();
  
  const question = "How many Employees are there?";
  console.log(`Question: ${question}`);
  
  const result = await generate_sql_query(question, db, llm);
  
  console.log(`Generated SQL: ${result.query}`);
  console.log('Query Results:');
  console.log(JSON.stringify(result.result, null, 2));
}

async function examples_of_langchain_db_llm() {
  const db = await get_db();
  const result = await db.run("SELECT * FROM Artist LIMIT 10;");
  console.log(result);

  const queryPromptTemplate = await pull<ChatPromptTemplate>(
    "langchain-ai/sql-query-system-prompt",
  );

  queryPromptTemplate.promptMessages.forEach((message) => {
    console.log(message.lc_kwargs.prompt.template);
  });

  const llm = get_llm();

  const queryOutput = z.object({
    query: z.string().describe("Syntactically valid SQL query."),
  });

  const structuredLlm = llm.withStructuredOutput(queryOutput);

  const writeQuery = async (state: typeof InputStateAnnotation.State) => {
    const promptValue = await queryPromptTemplate.invoke({
      dialect: db.appDataSourceOptions.type,
      top_k: 10,
      table_info: await db.getTableInfo(),
      input: state.question,
    });

    const result = await structuredLlm.invoke(promptValue);
    return { query: result.query };
  };

  const result_2 = await writeQuery({
    question: "How many Employees are there?",
  });
  console.log(result_2);

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
  let finalResult: any[] = [];
  for await (const step of await graph.stream(inputs, {
    streamMode: "updates",
  })) {
    console.log(step);
    console.log("\n====\n");
    finalResult.push(step);
  }
  console.log("Final Answer:");
  console.log(finalResult);
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////// Agent /////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

main();
