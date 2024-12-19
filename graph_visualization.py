from imports import *

def validate_partition(G, partition):
    """
     Validate the partition of a graph by checking for missing and extra nodes.

     Parameters:
     G (networkx.Graph): The graph to validate.
     partition (list of lists): The partition of the graph, where each sublist represents a community.

     Returns:
     tuple: A tuple containing two sets:
         - missing_nodes: Nodes present in the graph but missing in the partition.
         - extra_nodes: Nodes present in the partition but not in the graph.
     """
    # Flatten the partition to get all nodes in communities
    partition_nodes = set([node for community in partition for node in community])

    # Get all nodes in the graph
    graph_nodes = set(G.nodes())

    # Find missing and extra nodes
    missing_nodes = graph_nodes - partition_nodes
    extra_nodes = partition_nodes - graph_nodes

    if missing_nodes:
        print(f"Missing nodes from partition: {missing_nodes}")

    if extra_nodes:
        print(f"Extra nodes in partition: {extra_nodes}")

    return missing_nodes, extra_nodes


def clean_message(text):
    """
       Clean the message text by removing non-alphabetical characters, converting to lowercase, and removing stopwords.

       Parameters:
       text (str): The message text to clean.

       Returns:
       str: The cleaned message text.
       """
    # Remove non-alphabetical characters (except for some punctuation like commas, periods, etc.)
    text = re.sub(r"[^a-zA-Z0-9,.?!'\s]", "", text)

    # Convert text to lowercase
    text = text.lower()

    # Replace newlines and multiple spaces with a single space
    text = re.sub(r"[\s]+", " ", text).strip()
    # Define custom stopwords
    stopwords = set(['the', 'and', 'to', 'of', 'in', 'a', 'for', 'on', 'is', 'that', 'with',
                     'as', 'are', 'this', 'by', 'be', 'it', 'or', 'from', 'at', 'an', 'will', 'cc'])

    text = " ".join([word for word in text.split() if word.lower() not in stopwords])
    return text


def apply_community_detection(G, algorithm, next_slider):
    """
       Apply community detection algorithm to the graph.

       Parameters:
       G (networkx.Graph): The graph to apply community detection on.
       algorithm (str): The community detection algorithm to use ("Louvain" or "Girvan-Newman").
       next_slider (int): The number of partitions to retrieve for the Girvan-Newman algorithm.

       Returns:
       tuple: A tuple containing:
           - partition (dict): A dictionary mapping nodes to community labels.
           - modularity_score (float): The modularity score of the partition.
       """
    if algorithm == "Louvain":
        # Louvain outputs a dict where each node maps to a community label
        partition = community_louvain.best_partition(G.to_undirected())
        modularity_score = community_louvain.modularity(partition, G)

    elif algorithm == "Girvan-Newman":
        # Girvan-Newman returns an iterable of sets; we need to get the first partition
        communities_generator = nx_comm.girvan_newman(G)


        # Retrieve 'next_slider' partitions
        for i in range(next_slider):
            # Get the next partition (this is a tuple of sets)
            current_partition = next(communities_generator)  # Use a separate variable

            # Convert the tuple of sets into a dictionary mapping nodes to community indices
            partition = {node: i for i, comm in enumerate(current_partition) for node in comm}

            # Convert the communities into a list format
            other_partition = [list(community) for community in current_partition]

            # -----------------------------------------
            # Validate the partition
            missing_nodes, extra_nodes = validate_partition(G, other_partition)

            # Add missing nodes as singleton communities (if necessary)
            if missing_nodes:
                other_partition.extend([[node] for node in missing_nodes])

            # Remove extra nodes from the partition (if necessary)
            for community in other_partition:
                community[:] = [node for node in community if node in G.nodes()]

            # Ensure no node is in multiple communities
            other_partition = [list(set(community)) for community in other_partition]

            # ------------------------------------------
            # Calculate the modularity score for the partition
            modularity_score = modularity(G, other_partition)

    return partition, modularity_score


def generate_community_wordcloud(community, valid_dict):
    """
    Generate and display a word cloud for a given community.

    Parameters:
    community (list): List of nodes in the community.
    valid_dict (dict): Dictionary containing messages for each sender-recipient pair.
    """

    # Array that collects all people's messages to later create the
    mails_in_community = []
    # Collect messages from the community
    for sender in community:
        for recipient, messages in valid_dict[sender].items():
            if recipient in community and recipient != sender:
                for message in messages:
                    mails_in_community.append(clean_message(message))

    # Convert list to string and generate wordcloud
    unique_string = set(mails_in_community)

    unique_string = " ".join(unique_string)

    wordcloud = WordCloud(width=1000, height=500).generate(unique_string)

    # Plot wordcloud
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.imshow(wordcloud, interpolation="bilinear")

    ax.axis("off")

    # Display in Streamlit
    st.pyplot(fig)


# Function to generate and plot a dendrogram based on the graph's adjacency matrix
def generate_dendrogram(G, labels):
    """
     Generate and display a dendrogram based on the graph's adjacency matrix.

     Parameters:
     G (networkx.Graph): The graph to generate the dendrogram for.
     labels (list): List of labels for the nodes in the graph.
     """
    labels.pop()

    # Compute the adjacency matrix of the graph using nx.to_numpy_array
    adjacency_matrix = nx.to_numpy_array(G)

    # Perform hierarchical clustering
    Z = linkage(adjacency_matrix, 'ward')

    # Create a dendrogram using plotly
    fig = ff.create_dendrogram(Z)

    # Get the correct order of labels from the dendrogram's leaves
    dendro_leaves = (fig['layout']['xaxis']['tickvals'])

    # Ensure labels correspond to the number of leaves
    if len(labels) != len(dendro_leaves):

        # Handle mismatch between labels and leaves
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match the number of leaves ({len(dendro_leaves)}).")

    # Update the x-axis with correctly ordered labels
    fig.update_layout(width=1000, height=600)

    # Arranges the labels and leafs correctly
    fig.update_xaxes(ticktext=[labels[i] for i in np.argsort(dendro_leaves)], tickvals=dendro_leaves)

    st.plotly_chart(fig)


# Function to create and render the graph
def create_graph(connection_dict, threshold, subjects, distance, show_node_names, title, algorithm, messages_dict,
                 next_slider):
    """
     Create and render a graph based on the connection dictionary and other parameters.

     Parameters:
     connection_dict (dict): Dictionary containing connections between nodes.
     threshold (int): Minimum message threshold for including an edge.
     subjects (int): Number of top users to include in the graph.
     distance (float): Distance between nodes in the layout.
     show_node_names (bool): Whether to show node names in the graph.
     title (str): Title of the graph.
     algorithm (str): Community detection algorithm to use.
     messages_dict (dict): Dictionary containing messages for each sender-recipient pair.
     next_slider (int): Number of partitions to retrieve for the Girvan-Newman algorithm.
     """
    # Create a directed graph from user_connection_dict
    G = nx.DiGraph()

    # Add edges to the graph with weights, excluding "Unlisted Recipients"
    for user, connections in connection_dict.items():
        for recipient, weight in connections.items():

            # Avoids unlisted users - emails that we don't know their source user.
            if recipient != "Unlisted Recipients" and weight >= threshold:
                recipient_weight = connection_dict.get(recipient, {}).get(user, 0)
                if recipient_weight >= threshold:
                    G.add_edge(user, recipient, weight=min(weight, recipient_weight))

    # Calculate degree centrality to find influential users
    centrality = nx.degree_centrality(G)

    # Sort users by centrality and select the top users
    top_users = sorted(centrality, key=centrality.get, reverse=True)[:subjects]

    # Create a subgraph with only the top users
    G_top = G.subgraph(top_users)

    # Convert to a mutable graph
    G_top = nx.Graph(G_top)

    # Apply the selected community detection algorithm
    partition, modul_score = apply_community_detection(G_top, algorithm, next_slider)

    modul_score = format(modul_score, ".2f")

    st.markdown(f"<h3 style='text-align: center;'>Dendrogram Of Top {subjects} </h3>", unsafe_allow_html=True)
    generate_dendrogram(G_top, top_users)

    # Remove edges between different communities
    edges_to_remove = [(u, v) for u, v in G_top.edges() if partition[u] != partition[v]]
    G_top.remove_edges_from(edges_to_remove)

    # Get unique community assignments
    communities = set(partition.values())

    # Create a color map for communities
    colors = {community: f"rgb({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)})"
              for community in communities}

    # Assign positions with increased spacing for connected nodes
    node_pos = nx.spring_layout(G_top, dim=3, k=distance)

    # Extract node positions
    x_nodes = [node_pos[node][0] for node in G_top.nodes]
    y_nodes = [node_pos[node][1] for node in G_top.nodes]
    z_nodes = [node_pos[node][2] for node in G_top.nodes]

    # Create edge traces (including edges within communities)
    edge_trace = []
    for edge in G_top.edges():
        x0, y0, z0 = node_pos[edge[0]]
        x1, y1, z1 = node_pos[edge[1]]
        edge_trace.append(go.Scatter3d(
            x=[x0, x1, None], y=[y0, y1, None], z=[z0, z1, None],
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none'
        ))

    # Create node trace
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text' if show_node_names else 'markers',
        marker=dict(
            size=10,
            color=[partition[node] for node in G_top.nodes],  # Color by community
            colorscale='Viridis',
            opacity=0.8,
        ),
        text=list(G_top.nodes) if show_node_names else None,
        hoverinfo='text'
    )

    # Headlines
    st.markdown(f"<h3 style='text-align: center; color: white;'>{title}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: white;'>Modularity Score: {modul_score}</h4>",
                unsafe_allow_html=True)

    # Create the figure
    fig = go.Figure(data=edge_trace + [node_trace])

    # Set layout for 3D graph
    fig.update_layout(
        width=1000,
        height=600,
        title="",
        showlegend=False,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Show the interactive plot
    st.plotly_chart(fig)

    # Display the community color legend
    st.markdown(f"<h3 style='text-align: center; color: white;'>Graph's Legend</h3>", unsafe_allow_html=True)

    # Sets the maximum community
    max_community = []
    for community, color in colors.items():

        # Collects all the users that are under the same community
        current_community = [node for node in G_top.nodes if partition[node] == community]
        if len(current_community) > len(max_community):
            max_community = current_community

        # Prints the communities to the legend
        st.markdown(
            f"<span style='color: {color};'>{community}</span>: {', '.join([node for node in G_top.nodes if partition[node] == community])}",
            unsafe_allow_html=True)

    # Don't want communities of one person to have Wordcloud
    if len(max_community) <= 1:
        st.markdown("<h3 style='text-align: center; color: red;'>Communities Too Small</h3>", unsafe_allow_html=True)

    # Generating the appropriate Wordcloud for the biggest community found in the graph
    else:
        st.markdown("<h3 style='text-align: center;'>Word Cloud For The Biggest Community</h3>", unsafe_allow_html=True)
        people_in_community = ", ".join(max_community)
        st.markdown(f"<h5 style='text-align: center;'>{people_in_community}</h5>", unsafe_allow_html=True)
        generate_community_wordcloud(max_community, messages_dict)


def open_dict_assets():
    """
      Open and load the necessary dictionary assets from pickle files.

      Returns:
      tuple: A tuple containing four dictionaries:
          - user_connection_dict: Dictionary of user connections.
          - email_connection_dict: Dictionary of email connections.
          - user_messages_dict: Dictionary of user messages.
          - email_messages_dict: Dictionary of email messages.
      """
    user_connection_dict_dir = 'user_connection_dict_rd.pkl'
    email_connection_dict_dir = 'email_connection_dict_rd.pkl'

    user_messages_dict_dir = 'user_messages_dict_rd.pkl'
    email_messages_dict_dir = 'email_messages_dict_rd.pkl'

    # Get current directory
    current_directory = Path(__file__).parent

    # Load user connection dictionaries
    with open(os.path.join(current_directory, user_connection_dict_dir), 'rb') as user_pickle_file:
        user_connection_dict = pickle.load(user_pickle_file)

    with open(os.path.join(current_directory, email_connection_dict_dir), 'rb') as email_pickle_file:
        email_connection_dict = pickle.load(email_pickle_file)

    with open(os.path.join(current_directory, user_messages_dict_dir), 'rb') as user_pickle_file:
        user_messages_dict = pickle.load(user_pickle_file)

    with open(os.path.join(current_directory, email_messages_dict_dir), 'rb') as email_pickle_file:
        email_messages_dict = pickle.load(email_pickle_file)
    return user_connection_dict, email_connection_dict, user_messages_dict, email_messages_dict


if __name__ == '__main__':

    # Opens the necessary files to run the streamlit page
    user_connection_dict, email_connection_dict, user_messages_dict, email_messages_dict = open_dict_assets()

    # Streamlit app starts here
    st.markdown("<h1 style='text-align: center;'>Users Community Detection </h1>", unsafe_allow_html=True)

    # Dropdown for algorithm selection for Graph 1
    algorithm_1 = st.selectbox("Community Detection Algorithm for Users", [
        "Louvain",
        "Girvan-Newman"], key="algorithm_1")

    # Sets the base values depending on the chosen algorithm
    if algorithm_1 == "Louvain":
        max_val = 10000
        min_val = 2
        base_val = 5
        next_slider = 1
        max_user = 150
        base_user = 50

    else:
        max_val = 200
        min_val = 2
        base_val = 1
        next_slider = st.slider("Next Partition", 1, 10, 1, 1, key="next_slider")
        max_user = 30
        base_user = 15

    # Sliders for Graph 1
    threshold_1 = st.slider("Minimum Message Threshold", min_val, max_val, base_val, 1, key="threshold_1")
    subjects_1 = st.slider("Top Users Amount", 1, max_user, base_user, key="subjects_1")
    distance_1 = st.slider("Distance Between Nodes", 0.1, 8.0, 4.0, 0.1, key="distance_1")
    show_user_names = st.checkbox("Show Node Names", value=True, key="show_names_1")

    # Create the first graph
    create_graph(user_connection_dict, threshold_1, subjects_1, distance_1, show_user_names,
                 f"Graph 1 - Top {subjects_1} Users Communities", algorithm_1, user_messages_dict, next_slider)

    st.markdown("<h1 style='text-align: center;'>Emails Community Detection </h1>", unsafe_allow_html=True)

    # Dropdown for algorithm selection for Graph 2
    algorithm_2 = st.selectbox("Community Detection Algorithm for Emails", [
        "Louvain",
        "Girvan-Newman"], key="algorithm_2")

    # Sets the base values depending on the chosen algorithm
    if algorithm_2 == "Louvain":
        max_val = 10000
        min_val = 1
        base_val = 5
        next_slider = 1
        max_email = 150
        base_email = 15
    else:
        max_val = 500
        min_val = 1
        base_val = 100
        next_slider_2 = st.slider("Next Partition", 1, 10, 1, 1, key="next_slider_2")
        max_email = 30
        base_email = 15

    # Sliders for Graph 2
    threshold_2 = st.slider("Minimum Message Threshold", min_val, max_val, base_val, 1, key="threshold_2")
    subjects_2 = st.slider("Top Emails Amount", 1, max_email, base_email, key="subjects_2")
    distance_2 = st.slider("Distance Between Nodes", 0.1, 4.0, 1.5, 0.1, key="distance_2")
    show_email_names = st.checkbox("Show Node Names", value=True, key="show_names_2")

    # Create the second graph
    create_graph(email_connection_dict, threshold_2, subjects_2, distance_2, show_email_names,
                 f"Graph 2 - Top {subjects_2} Emails Communities", algorithm_2, email_messages_dict, next_slider)
