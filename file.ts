import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";

async function main() {
  const datasource = new DataSource({
    type: "sqlite",
    database: "Chinook.db",
  });
  const db = await SqlDatabase.fromDataSourceParams({
    appDataSource: datasource,
  });

  const result = await db.run("SELECT * FROM Artist LIMIT 10;");
  console.log(result);
}

main().catch(console.error);
