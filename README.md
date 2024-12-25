
# **Enron Project**  
_Analyzing Enron's email dataset and stock prices to uncover insights through sentiment analysis, network analysis, and data visualization._

## **Table of Contents**
1. [About the Project](#about-the-project)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Getting Started](#getting-started)
5. [Usage](#usage)
6. [Scripts and Notebooks](#scripts-and-notebooks)
7. [Visualization](#visualization)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

---

## **About the Project**
This project leverages the Enron email dataset and stock price data to explore the relationships between communication patterns and financial trends. Key tasks include:
- Sentiment analysis of emails.
- Network analysis to identify key individuals and their influence.
- Correlation analysis between sentiment trends and stock performance.

The project includes visualizations and scripts to extract meaningful insights, enabling users to better understand how communication and sentiment might impact stock trends.

---

## **Features**
- **Sentiment Analysis**: Analyzes sentiment trends over time using email data.
- **Network Analysis**: Identifies key individuals within the Enron network using PageRank and centrality measures.
- **Stock Price Correlation**: Examines relationships between email sentiment and Enron stock prices.
- **Graph Visualizations**: Creates interactive visualizations for email networks.

---

## **Dataset**
- **Email Data**: Processed email communications from Enron employees.
- **Stock Data**: Historical Enron stock prices provided in `enron_stock_price.csv`.

---

## **Getting Started**
### **Prerequisites**
Ensure you have the following installed:
- Python 3.8+
- Jupyter Notebook
- Required Python libraries (see below)

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Enron-project.git
   cd Enron-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
### **Run the Analysis**
1. Open the main Jupyter notebook:
   ```bash
   jupyter notebook ENRON_PROJECT.ipynb
   ```
2. Run the notebook cells to:
   - Perform sentiment analysis.
   - Analyze the network graph.
   - Visualize the results.

### **Scripts**
- **`imports.py`**: Handles library imports and reusable functions.
- **`mailClass.py`**: Contains the logic for email data parsing and processing.
- **`graph_visualization.py`**: Generates network graph visualizations.

---

## **Scripts and Notebooks**
- **`ENRON_PROJECT.ipynb`**: The primary notebook for executing the entire analysis pipeline.
- **`graph_visualization.py`**: A script dedicated to generating interactive graph visualizations for the network.

---

## **Visualization**
This project includes visualizations for:
1. **Network Graphs**: Interactive visualizations highlighting key individuals and their influence within the Enron network.
2. **Sentiment Trends**: Charts showcasing sentiment changes over time.
3. **Stock-Sentiment Correlation**: Graphs comparing sentiment scores with stock prices.

---


## **Contact**
**Your Name**  
- Email: [mikelmessika@gmail.com](mailto:your_email@example.com)  
- GitHub: [michael7070653](https://github.com/your_username)

---
