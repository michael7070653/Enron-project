import re
import email
import pickle
import numpy as np
import email.utils
import pandas as pd
import networkx as nx
import streamlit as st
from typing import List
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from collections import defaultdict
import community as community_louvain
from networkx.algorithms import community as nx_comm
from scipy.cluster.hierarchy import linkage, dendrogram
from pathlib import Path
import os
from networkx.algorithms.community import modularity


