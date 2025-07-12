import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";
import { Annotation } from "@langchain/langgraph";
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
import { ChatAnthropic } from "@langchain/anthropic";

import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";

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
}

main().catch(console.error);
