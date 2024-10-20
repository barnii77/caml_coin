import sqlite3
import cc_utils

DATABASE_PATH = "users.db"


def create_users_table():
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                private_key BLOB NOT NULL,
                public_key BLOB NOT NULL
            )
        """
        )
        conn.commit()


def insert_user(user_id, private_key, public_key):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (user_id, private_key, public_key) 
            VALUES (?, ?, ?)
        """,
            (user_id, private_key, public_key),
        )
        conn.commit()


def get_user_keys(user_id):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT private_key, public_key 
            FROM users 
            WHERE user_id = ?
        """,
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            return {"private_key": row[0], "public_key": row[1]}
        else:
            return None


class CamlCoinUserInfo:
    def __init__(self, user_id, private_key: bytes, public_key: bytes):
        if len(public_key) != cc_utils.PUBLIC_KEY_SIZE or len(private_key) != cc_utils.PRIVATE_KEY_SIZE:
            raise ValueError(
                f"public key and private key must be 33 bytes and 32 bytes in length respectively, but are {len(public_key)} and {len(private_key)} bytes long"
            )
        self.user_id = user_id
        self.private_key = private_key
        self.public_key = public_key

    def store_to_db(self):
        insert_user(self.user_id, self.private_key, self.public_key)


def get_user_info(user_id):
    keys = get_user_keys(user_id)
    return CamlCoinUserInfo(user_id, keys["private_key"], keys["public_key"])


if __name__ == "__main__":
    create_users_table()
