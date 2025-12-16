#include <vector>
#include <list>

using namespace std;

#define MAX_SORTS 20
// Algorithm from: https://www.geeksforgeeks.org/all-topological-sorts-of-a-directed-acyclic-graph/#

template<typename T>
class Graph {
    vector<T> vertices;
    int V;    // No. of vertices

    // Pointer to an array containing adjacency list
    list<T> *adj;

    // Vector to store indegree of vertices
    vector<int> indegree;

    // A function used by alltopologicalSort
    void allTopologicalSortUtil(vector<vector<T>> &result, vector<int> &res, bool *visited);

    // A function used by isCyclic
    bool isCyclicUtil(int src, vector<bool> &visited, vector<bool> &recursiveStack);


public:
    // Graph(int V);   // Constructor
    explicit Graph(vector<T> vertices);   // Constructor

    // function to add an edge to graph
    void addEdge(pair<T, T> edge);

    void addEdge(T src, T dst);

    // funciton to add in all arcs (list of edges)
    void addArcs(vector<pair<T, T>> arcs);

    int getVertexIndex(T v);

    // Prints all Topological Sorts
    vector<vector<T>> allTopologicalSort();

    bool isCyclic();

    void print();

    void printAllOrders(vector<vector<T>> allOrders);

};

template<typename T>
Graph<T>::Graph(vector<T> vertices) {
    this->vertices = vertices;
    this->V = (int) vertices.size();

    this->adj = new list<T>[V];

    // Initialize all indegrees with 0
    for (int i = 0; i < V; i++) {
        indegree.push_back(0);
    }
}


template<typename T>
int Graph<T>::getVertexIndex(T v) {
    auto it = find(vertices.begin(), vertices.end(), v);

    if (it != vertices.end()) {
        return it - vertices.begin();
    } else {
        return -1;
    }
}

template<typename T>
void Graph<T>::addArcs(vector<pair<T, T>> arcs) {
    for (auto arc: arcs) {
        addEdge(arc);
    }
}

template<typename T>
void Graph<T>::addEdge(pair<T, T> edge) {
    auto src = edge.first;
    auto i_src = getVertexIndex(src);
    auto dst = edge.second;
    auto i_dst = getVertexIndex(dst);

    // Ignore arcs for vertices that cannot be found
    if (i_src != -1 and i_dst != -1) {
        adj[i_src].push_back(i_dst);
        indegree[i_dst]++;
    }


}

template<typename T>
void Graph<T>::addEdge(T src, T dst) {
    auto i_src = getVertexIndex(src);
    auto i_dst = getVertexIndex(dst);

    adj[i_src].push_back(i_dst);

    indegree[i_dst]++;
}

// TODO (owhsu): Topological sort will always return the first MAX_SORTS starting given the order of the vertices.
//               Future work should also pass in an initial permutation of vertices
//               such that we can explore the dataflow space more. This is not important for now.
template<typename T>
void Graph<T>::allTopologicalSortUtil(vector<vector<T>> &result, vector<int> &res,
                                      bool visited[]) {
    if (result.size() > MAX_SORTS) {
        return;
    }

    // To indicate whether all topological are found or not
    bool flag = false;

    for (int i = 0; i < V; i++) {
        //  If indegree is 0 and not yet visited then only choose that vertex
        if (indegree[i] == 0 && !visited[i]) {
            //  reducing indegree of adjacent vertices
            list<int>::iterator j;
            for (j = adj[i].begin(); j != adj[i].end(); j++)
                indegree[*j]--;

            //  including in result
            res.push_back(i);
            visited[i] = true;
            allTopologicalSortUtil(result, res, visited);

            // resetting visited, res and indegree for
            // backtracking
            visited[i] = false;
            res.erase(res.end() - 1);
            for (j = adj[i].begin(); j != adj[i].end(); j++)
                indegree[*j]++;

            flag = true;
        }
    }

    //  We reach here if all vertices are visited. So we add one of the orders to the final result here
    if (!flag) {
        vector<T> temp;
        for (auto index: res) {
            temp.push_back(vertices.at(index));
            // cout << vertices.at(index) << " ";
        }
        // cout << endl;
        result.push_back(temp);
    }
}

//  The function does all Topological Sort. It uses recursive allTopologicalSortUtil(...)
template<typename T>
vector<vector<T>> Graph<T>::allTopologicalSort() {

    // Mark all the vertices as not visited
    bool *visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;

    vector<vector<T>> result;
    vector<int> res;
    allTopologicalSortUtil(result, res, visited);
    return result;
}

template<typename T>
bool Graph<T>::isCyclic() {
    vector<bool> visited(V, false);
    vector<bool> recursiveStack(V, false);

    for (int i = 0; i < V; i++) {
        if (!visited[i] && isCyclicUtil(i, visited, recursiveStack))
            return true;
    }
    return false;
}

template<typename T>
bool Graph<T>::isCyclicUtil(int src, vector<bool> &visited, vector<bool> &recursiveStack) {
    if (!visited[src]) {

        // Mark the current node as visited and part of recursion stack
        visited[src] = true;
        recursiveStack[src] = true;

        // Recur for all the vertices adjacent to this vertex
        for (int dst: adj[src]) {
            if (!visited[dst] && isCyclicUtil(dst, visited, recursiveStack))
                return true;
            else if (recursiveStack[dst])
                return true;
        }
    }

    // Remove the vertex from recursion stack
    recursiveStack[src] = false;
    return false;
}

// Function to nicely print out graph information
template<typename T>
void Graph<T>::print() {
    cout << "Begin TopoSort Graph" << endl;
    cout << "Node list: ";
    for (auto v: vertices) {
        cout << v << ", ";
    }
    cout << endl;

    cout << "Edges: ";
    for (int i = 0; i < V; i++) {
        cout << "edge " << i << ": ";
        for (auto it = adj[i].begin(); it != adj[i].end(); ++it) {
            cout << vertices.at(i) << "->" << *it << ", ";
        }
        cout << endl;
    }
    cout << "End TopoSort Graph" << endl;
}



// TODO: add function to find cycyles

//template <typename T>
//bool getAllTopologicalSorts(const std::vector<T>& nodes,
//                            const std::vector<std::pair<T, T>>& arcs,
//                            std::vector<T>* topological_order) {
//  return internal::TopologicalSortImpl<T, false>(nodes, arcs, topological_order,
//                                                 nullptr);
//}
