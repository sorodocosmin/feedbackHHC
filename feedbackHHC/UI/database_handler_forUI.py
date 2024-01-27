import psycopg2

class Config:
    DATABASE_NAME = 'HomeCare'
    DATABASE_USER = 'postgres'
    DATABASE_PASSWORD = '123456'
    DATABASE_HOST = 'localhost'
    DATABASE_PORT = '5432'

class DatabaseHandler:
    @staticmethod
    def create_connection(db_name, db_user, db_password, db_host, db_port):
        connection = None

        try:
            connection = psycopg2.connect(
                database=db_name,
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port
            )
            # print("Connection to PostgreSQL DB successful")
        except Exception as e:
            print(f"The error '{e}' occurred")
    
        return connection
    