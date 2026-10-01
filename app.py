from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from database import get_transactions, create_database
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlite3
import joblib


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        if "statement" not in request.files:
            return "No file selected."

        file = request.files["statement"]

        if file.filename == "":
            return "Please choose a CSV file."

        if not file.filename.lower().endswith(".csv"):
            return "Only CSV files are allowed."

        try:
            upload_folder = os.path.join(BASE_DIR, "uploads")
            os.makedirs(upload_folder, exist_ok=True)

            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            required_columns = [
                "Date",
                "Description",
                "Debit",
                "Credit",
                "Balance"
            ]

            if df.empty:
                return "The uploaded CSV file is empty."

            for column in required_columns:
                if column not in df.columns:
                    return f"Missing required column: {column}"

            X = vectorizer.transform(df["Description"])
            predicted_categories = model.predict(X)
            df["Predicted Category"] = predicted_categories

            category_summary = (
                df["Predicted Category"]
                .value_counts()
                .to_dict()
            )

            conn = sqlite3.connect(
                os.path.join(BASE_DIR, "database", "bank.db")
            )
            cursor = conn.cursor()

            cursor.execute("DELETE FROM transactions")

            for _, row in df.iterrows():
                cursor.execute(
                    """
                    INSERT INTO transactions
                    (date, description, debit, credit, balance, category)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["Date"],
                        row["Description"],
                        row["Debit"],
                        row["Credit"],
                        row["Balance"],
                        row["Predicted Category"]
                    )
                )

            conn.commit()
            conn.close()

            total_income = df["Credit"].sum()
            total_expense = df["Debit"].sum()
            current_balance = df["Balance"].iloc[-1]
            total_transactions = len(df)

            highest_income = df.loc[df["Credit"].idxmax()]
            highest_expense = df.loc[df["Debit"].idxmax()]

            top_category = (
                df["Predicted Category"]
                .value_counts()
                .idxmax()
            )

            income_transactions = (df["Credit"] > 0).sum()
            expense_transactions = (df["Debit"] > 0).sum()

            plt.figure(figsize=(5, 5))

            plt.pie(
                [total_income, total_expense],
                labels=["Income", "Expense"],
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize" : 16}
            )

            plt.savefig(
                os.path.join(BASE_DIR, "static", "images", "pie_chart.png")
            )
            plt.close()

            plt.figure(figsize=(5, 5))

            plt.bar(
               ["Income", "Expense"],
               [total_income, total_expense],
               color=["green", "red"]
            )

            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel("Amount (₹)", fontsize=12)
            plt.savefig(
                os.path.join(BASE_DIR, "static", "images", "bar_chart.png")
            )
            plt.close()

            table = df.to_html(classes="table", index=False)

            return render_template(
                "dashboard.html",
                table=table,
                income=total_income,
                expense=total_expense,
                balance=current_balance,
                transactions=total_transactions,
                category_summary=category_summary,
                top_category=top_category,
                highest_expense=highest_expense,
                highest_income=highest_income,
                income_transactions=income_transactions,
                expense_transactions=expense_transactions
            )

        except Exception as e:
            return f"Error: {e}"

    return render_template("upload.html")

@app.route("/dashboard")
def dashboard():

    df = get_transactions()

    search = request.args.get("search")
    transaction_type = request.args.get("type")

    if search:
        df = df[
            df["description"].astype(str).str.contains(
                search,
                case=False,
                na=False
            )
        ]

    if transaction_type == "income":
        df = df[df["credit"] > 0]

    elif transaction_type == "expense":
        df = df[df["debit"] > 0]

    table = df.to_html(classes="table", index=False)

    income = df["credit"].sum()
    expense = df["debit"].sum()
    transactions = len(df)

    if not df.empty:
        balance = df["balance"].iloc[-1]
        highest_income = df.loc[df["credit"].idxmax()]
        highest_expense = df.loc[df["debit"].idxmax()]

        if "category" in df.columns and not df["category"].dropna().empty:
            top_category = df["category"].value_counts().idxmax()
        else:
            top_category = "N/A"

    else:
        balance = 0
        highest_income = None
        highest_expense = None
        top_category = "N/A"

    return render_template(
        "dashboard.html",
        table=table,
        income=income,
        expense=expense,
        balance=balance,
        transactions=transactions,
        highest_income=highest_income,
        highest_expense=highest_expense,
        top_category=top_category
    )

@app.route("/download")
def download():

    df = get_transactions()

    search = request.args.get("search")
    transaction_type = request.args.get("type")

    if search:
        df = df[
            df["description"].astype(str).str.contains(
                search,
                case=False,
                na=False
            )
        ]

    if transaction_type == "income":
        df = df[df["credit"] > 0]

    elif transaction_type == "expense":
        df = df[df["debit"] > 0]

    file_path = "filtered_transactions.csv"
    df.to_csv(file_path, index=False)

    return send_file(file_path, as_attachment=True)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/transactions")
def transactions():

    df = get_transactions()

    search = request.args.get("search")
    transaction_type = request.args.get("type")

    if search:
        df = df[
            df["description"].astype(str).str.contains(
                search,
                case=False,
                na=False
            )
        ]

    if transaction_type == "income":
        df = df[df["credit"] > 0]

    elif transaction_type == "expense":
        df = df[df["debit"] > 0]

    if df.empty:
        table = "<h3 style='text-align:center;'>No transaction found.</h3>"
    else:
        table = df.to_html(classes="table", index=False)

    return render_template(
        "transactions.html",
        table=table
    )

if __name__ == "__main__":
    create_database()
    app.run()