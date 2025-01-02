import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import streamlit as st

# Function to load and clean data
def loadcleandata(file_path):
    record = pd.read_csv(file_path)
    columns = [
        "Format", "Total Matches Played", "England Wins", "England Losses",
        "Australia Wins", "Australia Losses", "Ties/Draws", "No Result",
        "England Win %", "Australia Win %", "Top Batter", "Top Bowler",
        "Aus Avg vs Eng Pace", "Aus Avg vs Eng Spin",
        "Eng Avg vs Aus Pace", "Eng Avg vs Aus Spin"
    ]
    record = record.iloc[2:5].reset_index(drop=True)
    record.columns = columns
    # Converting datatypes from string to float using rstrip for removal too
    record["England Win %"] = pd.to_numeric(record["England Win %"].astype(str).str.rstrip('%').astype(float))
    record["Australia Win %"] = pd.to_numeric(record["Australia Win %"].astype(str).str.rstrip('%').astype(float))
    return record

# Taking Toss input for user that which team will win the toss
def input_toss():
    toss = st.selectbox("Select Toss Winner", ['England', 'Australia'])
    return toss

# Prediction using Linear Regression model
def prediction(record, toss_winner):
    en = LabelEncoder()  # This will convert text or any data to numeric value
    record["Format Encoded"] = en.fit_transform(record["Format"])  # Used for fitting data and transforming it

    S = record[["Format Encoded"]]
    z_england = record["England Win %"]
    z_australia = record["Australia Win %"]
    models_eng = LinearRegression().fit(S, z_england)
    models_aus = LinearRegression().fit(S, z_australia)
    record["Predicted England Win %"] = models_eng.predict(S)
    record["Predicted Australia Win %"] = models_aus.predict(S)
    
    # Calculate Average Squared difference between actual vs predicted 
    st.write("England MSE:", mean_squared_error(z_england, record["Predicted England Win %"]))
    st.write("Australia MSE:", mean_squared_error(z_australia, record["Predicted Australia Win %"]))
    
    if toss_winner == "England":
        record["England Win %"] += 5  # it's just a simple assumption made to represent the advantage
    elif toss_winner == "Australia":
        record["Australia Win %"] += 5

    return record, models_eng, models_aus, en

# Predicts the win probabilities for both England and Australia based on the given match format
def matchs_format(models_eng, models_aus, en, match_format):
    format = en.transform([match_format])[0]
    format_encoded_df = pd.DataFrame([[format]], columns=["Format Encoded"])
    win_england = models_eng.predict(format_encoded_df)[0]
    win_australia = models_aus.predict(format_encoded_df)[0]
    return win_england, win_australia

# Visualization: Actual vs Predicted
def predictions_visualizing(record):
    plt.figure(figsize=(10, 6))
    plt.plot(record["Format"], record["England Win %"], label="England Actual", marker="o", color="blue")
    plt.plot(record["Format"], record["Predicted England Win %"], label="England Predicted", linestyle="--", color="black")
    plt.plot(record["Format"], record["Australia Win %"], label="Australia Actual", marker="o", color="red")
    plt.plot(record["Format"], record["Predicted Australia Win %"], label="Australia Predicted", linestyle="--", color="yellow")

    for i in range(len(record)):
        if record["Predicted England Win %"][i] > record["Predicted Australia Win %"][i]:
            plt.text(record["Format"][i], record["Predicted England Win %"][i], "England", color="red", ha='center', va='bottom', weight='bold')
        else:
            plt.text(record["Format"][i], record["Predicted Australia Win %"][i], "Australia", color="red", ha='center', va='bottom', weight='bold')

    plt.title("Actual vs Predicted Win Percentages with Winning Predictions")
    plt.xlabel("Match Format")
    plt.ylabel("Win Percentage")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Use Streamlit's function to show the plot

# New Function: Visualize Win Probabilities as Bar Chart
def win_probability_visualize(record):
    teams = ['England', 'Australia']
    probability_winning = [
        record["Predicted England Win %"].mean(),
        record["Predicted Australia Win %"].mean()
    ]
    colors = ['#FF9999', '#90EE90']  # Red and green shades
    plt.figure(figsize=(10, 4))
    bars = plt.barh(teams, probability_winning, color=colors, edgecolor='black', height=0.6)
    for bar, probability in zip(bars, probability_winning):
        plt.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
                 f"{probability:.2f}%", ha='center', va='center', color='blue', fontsize=12, weight='bold')

    plt.title('Win Probability', fontsize=14, pad=20)
    plt.xlim(0, 100)
    plt.xlabel('Probability (%)', fontsize=12)
    plt.gca().invert_yaxis()  # Used in horizontal charts
    plt.box(False)
    plt.xticks(range(0, 101, 20))
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)  # Use Streamlit's function to show the plot

# Main function
def main():
    st.title("England vs Australia Match Prediction")
    
    # Load and clean data
    file_path = "australia vs england.csv"
    data = loadcleandata(file_path)
    
    toss_winner = input_toss()
    
    # Train linear regression models
    data, models_eng, models_aus, en = prediction(data, toss_winner)

    # Get user input for match format
    match_format = st.selectbox("Enter the Match Format", ["T20", "ODI", "Test"])

    # Predict win probabilities for the chosen format
    win_england, win_australia = matchs_format(models_eng, models_aus, en, match_format)

    st.write(f"Predicted England Win Probability: {win_england:.2f}%")
    st.write(f"Predicted Australia Win Probability: {win_australia:.2f}%")

    # Visualize predictions
    predictions_visualizing(data)
    win_probability_visualize(data)

if __name__ == "__main__":
    main()
