import { ChatAnthropic } from "@langchain/anthropic";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { SqlDatabase } from "langchain/sql_db";
import { QuerySqlTool } from "langchain/tools/sql";
import { DataSource } from "typeorm";
import { z } from "zod";

const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  query: Annotation<string>,
  result: Annotation<string>,
  answer: Annotation<string>,
});

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";

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
async function main() {
  // Query the database
  const datasource = new DataSource({
    type: "sqlite",
    database: "Chinook.db",
  });
  const db = await SqlDatabase.fromDataSourceParams({
    appDataSource: datasource,
  });

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

//
main().catch(console.error);
