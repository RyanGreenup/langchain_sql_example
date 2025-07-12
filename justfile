init_data:
    # https://js.langchain.com/docs/tutorials/sql_qa/
    curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db
