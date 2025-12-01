# type: ignore
import os
import pandas as pd  # type: ignore
import sqlalchemy as sa  # type: ignore
from sqlalchemy.exc import SQLAlchemyError  # type: ignore
from sqlalchemy.engine import URL
from dotenv import load_dotenv

load_dotenv()  # Lädt Umgebungsvariablen aus .env

class Database:
    def __init__(self):
        self._engine = None
        self._connection = None
        self._connect()

    def _connect(self):
        """
        Stellt die Verbindung zur PostgreSQL-Datenbank her.
        Die Zugangsdaten werden aus Umgebungsvariablen geladen.
        """
        try:
            db_url = URL.create(
                drivername="postgresql",
                username=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                query={"sslmode": os.getenv("DB_SSLMODE", "require")}
            )
            self._engine = sa.create_engine(db_url, pool_pre_ping=True)
            self._connection = self._engine.connect().execution_options(isolation_level="AUTOCOMMIT")
            print("✅ Verbindung zur PostgreSQL-Datenbank hergestellt.")
        except SQLAlchemyError as e:
            print(f"❌ Verbindungsfehler: {e}")
            self._engine = None
            self._connection = None

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Führt eine SQL-Abfrage aus und gibt das Ergebnis als DataFrame zurück.
        """
        if not self._connection:
            raise ConnectionError("⚠️ Keine aktive Datenbankverbindung.")
        try:
            df = pd.read_sql(sa.text(query), self._connection)
            print(f"✅ Abfrage erfolgreich. {len(df)} Zeilen abgerufen.")
            return df
        except SQLAlchemyError as e:
            print(f"❌ Abfragefehler: {e}")
            return pd.DataFrame()

    def execute_sql_file(self, sql_file_path: str) -> pd.DataFrame:
        """
        Führt eine SQL-Datei aus und gibt das Ergebnis als DataFrame zurück.
        """
        if not os.path.exists(sql_file_path):
            raise FileNotFoundError(f"⚠️ SQL-Datei nicht gefunden: {sql_file_path}")
        with open(sql_file_path, "r") as file:
            sql_query = file.read()
        print(f"📄 SQL-Datei wird ausgeführt: {sql_file_path}")
        return self.execute_query(sql_query)

    def close(self):
        """
        Schließt die Datenbankverbindung.
        """
        if self._connection:
            self._connection.close()
            print("🔒 Verbindung geschlossen.")
