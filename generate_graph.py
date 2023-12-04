import networkx as nx
import matplotlib.pyplot as plt

def get_graph():
    dag = nx.DiGraph()
    dag.add_nodes_from(['persuation', 'ethos', 'pathos', 'logos', 'ad hominem', 'bandwagon',
                        'appeal to authority', 'glitering generelalities', 'exaggeration',
                        'loaded language', 'flag waving', 'appeal to fear', 'justification',
                        'reasoning', 'repetition', 'intentional vagueness', 'name calling',
                        'doubt', 'smears', 'reductio ad hitlerum', 'whataboutism', 'slogans',
                        'distraction', 'simplification', 'straw man', 'red herring',
                        'casual oversimplification', 'black&white fallacy',
                        'thought terminating cliche'])

    dag.add_edges_from([('persuation', 'ethos'), ('persuation', 'pathos'), ('persuation', 'logos'),
                        ('ethos', 'ad hominem'), ('ethos', 'bandwagon'),
                        ('ethos', 'appeal to authority'), ('ethos', 'glitering generelalities'),
                        ('pathos', 'exaggeration'), ('pathos', 'loaded language'),
                        ('pathos', 'flag waving'), ('pathos', 'appeal to fear'),
                        ('logos', 'justification'), ('logos', 'reasoning'), ('logos', 'repetition'),
                        ('logos', 'intentional vagueness'),
                        ('ad hominem', 'name calling'), ('ad hominem', 'doubt'),
                        ('ad hominem', 'smears'), ('ad hominem', 'reductio ad hitlerum'),
                        ('ad hominem', 'whataboutism'),
                        ('justification', 'bandwagon'), ('justification', 'appeal to authority'),
                        ('justification', 'flag waving'), ('justification', 'appeal to fear'),
                        ('justification', 'slogans'),
                        ('reasoning', 'distraction'), ('reasoning', 'simplification'),
                        ('distraction', 'straw man'), ('distraction', 'red herring'),
                        ('distraction', 'whataboutism'),
                        ('simplification', 'casual oversimplification'),
                        ('simplification', 'black&white fallacy'),
                        ('simplification', 'thought terminating cliche')])

    pos = nx.spring_layout(dag)  # Positions for all nodes
    nx.draw(dag, pos, with_labels=True, arrowsize=20, node_size=2000, font_size=5,
            font_color="black", node_color="skyblue", font_weight="bold")

    # Show the graph
    plt.show()
    return dag