import os

from app import app, create_table


if __name__ == "__main__":
    create_table()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=False, use_reloader=False)
