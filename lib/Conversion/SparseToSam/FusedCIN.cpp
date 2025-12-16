#include "FusedCIN.h"
#include "Table.h"
#include "ortools/base/types.h"


// #include "ortools/graph/connected_components.h"
// #include "ortools/graph/dense_connected_component_finder.h"

unsigned int TensorViewImpl::counter = 0;

// TensorView llvm::DenseMapInfo<TensorView, void>::getEmptyKey() { return {}; }



TensorView llvm::DenseMapInfo<TensorView, void>::getTombstoneKey() {
    return TensorView(std::vector<UniqueVar>(), nullptr);
}

unsigned llvm::DenseMapInfo<TensorView>::getHashValue(const TensorView &val) { return val.getId() * 37U; }

bool llvm::DenseMapInfo<TensorView>::isEqual(const TensorView &LHS, const TensorView &RHS) { return LHS == RHS; }

UniqueVar AnalysisScope::getNewUniqueVar(const std::shared_ptr<FusedCIN> &tensor) {
    auto newVar = UniqueVar(tensor, indexCounter++);
    vars.push_back(newVar);
    return newVar;
}

void AnalysisScope::markEqual(const UniqueVar &a, const UniqueVar &b) {
    equalityGraph.AddArc(a.getId(), b.getId());
    equalityGraphComponents.SetNumberOfNodes(equalityGraph.num_nodes());
    equalityGraphComponents.AddEdge(a.getId(), b.getId());
}

void AnalysisScope::printConnectedComponents() {
    std::map<unsigned int, std::vector<unsigned int>> equalVars;
    for (auto componentIter: llvm::enumerate(equalityGraphComponents.GetComponentIds())) {
        const auto component = componentIter.value();
        const unsigned int componentId = componentIter.index();
        equalVars[component].push_back(componentId);
    }
    for (const auto &elem: equalVars) {
        llvm::outs() << "Element: " << elem.first << "\n";
        for (const auto item: elem.second) {
            llvm::outs() << "Connected: " << item << "\n";
        }
    }
}

IndexVar AnalysisScope::getLogicalIndexVar(const UniqueVar *var) {
    // std::cout << "Getting logical var for : " << *var << "\n";
    int root = equalityGraphComponents.FindRoot(var->getId());
    const auto it = logicalIndexMap.find(root);
    int newIndex = 0;
    if (it != logicalIndexMap.end()) {
        // Key exists, retrieve the value
        newIndex = it->second;
    } else {
        newIndex = logicalIndexCounter++;
        // Key does not exist, insert the new value
        logicalIndexMap.insert(std::pair(root, newIndex));
    }
    return IndexVar(newIndex);
}

void AnalysisScope::addPartialOrder(const UniqueVar &a, const UniqueVar &b) {
    partialOrderings.AddArc(b.getId(), a.getId());
}

void AnalysisScope::addNode(const UniqueVar &a) { partialOrderings.AddNode(a.getId()); }

std::vector<std::vector<IndexVar>> AnalysisScope::getAllLoopOrders() {
    std::vector<std::vector<IndexVar>> allOrdersIndexVars;
    std::vector<std::pair<int, int>> logicalIndexArcs;
    std::vector<int> resultSort;
    util::Graph IndexVarOrdering;
    for (auto node: partialOrderings.AllNodes()) {
        UniqueVar var = UniqueVar(node);
        if (llvm::find(IndexVarOrdering.AllNodes(), this->getLogicalIndexVar(&var).getId()) ==
            IndexVarOrdering.AllNodes().end()) {
            IndexVarOrdering.AddNode(this->getLogicalIndexVar(&var).getId());
        }
    }
    for (int i = 0; i < partialOrderings.num_arcs(); ++i) {
        UniqueVar newHeadVar = UniqueVar(partialOrderings.Head(i));
        UniqueVar newTailVar = UniqueVar(partialOrderings.Tail(i));
        IndexVarOrdering.AddArc(this->getLogicalIndexVar(&newTailVar).getId(),
                                this->getLogicalIndexVar(&newHeadVar).getId());
    }
    for (int i = 0; i < IndexVarOrdering.num_arcs(); ++i) {
        IndexVar newHeadVar = IndexVar(IndexVarOrdering.Head(i));
        IndexVar newTailVar = IndexVar(IndexVarOrdering.Tail(i));
        logicalIndexArcs.push_back({newHeadVar.getId(), newTailVar.getId()});
        // std::cout << "Arc: " << logicalIndexArcs[i].first << " -> "
        //           << logicalIndexArcs[i].second << "\n";
    }
    auto vec = IndexVarOrdering.AllNodes();
    std::vector<int> nodes;
    for (auto elem: vec) {
        // llvm::outs() << "ELEM: " << elem << "\n";
        nodes.push_back(elem);
    }

    std::vector<IndexVar> resultOrder;

    Graph<int> graph(nodes);
    graph.addArcs(logicalIndexArcs);
    // For debugging sorts
    // graph.print();
    auto cyclesFound = graph.isCyclic();

    if (!cyclesFound) {
        auto allOrdersInts = graph.allTopologicalSort();
        for (const auto &orderInts: allOrdersInts) {
            vector<IndexVar> orderIndexVars;
            for (auto elem: orderInts) {
                orderIndexVars.emplace_back(elem);
            }
            allOrdersIndexVars.push_back(orderIndexVars);
        }

        if (allOrdersInts.empty()) {
            std::cerr << "Cycle not found, but no valid loop orders found" << std::endl;
            exit(1);
        }
    } else {
        std::cerr << "Cycle found, invalid loop order provided" << std::endl;
        exit(1);
    }
    return allOrdersIndexVars;
}

std::vector<IndexVar> AnalysisScope::getLoopOrder() {
    std::vector<IndexVar> resultOrder;
    auto allOrdersIndexVars = getAllLoopOrders();

    std::cout << "// # of possible loop orders: " << allOrdersIndexVars.size() << std::endl;

    // FIXME (owhsu): Currently just get first result of all valid orders
    int selectOrder = 0;
    if (useUserInput) {
        printAllOrders(allOrdersIndexVars);
        std::cout << "Select which order between [0-" << (allOrdersIndexVars.size() - 1) << "]: ";
        std::cin >> selectOrder;
    }
    resultOrder = allOrdersIndexVars.at(selectOrder);

    return resultOrder;
}

IndexVar UniqueVar::getLogicalIndexVar() const { return tensor->scope->getLogicalIndexVar(this); }

std::ostream &operator<<(std::ostream &os, const UniqueVar &var) {
    os << "U" << var.getId();
    return os;
}

std::ostream &operator<<(std::ostream &os, const IndexVar &var) {
    os << "i" << var.getId();
    return os;
}

std::ostream &operator<<(std::ostream &os, const TensorViewImpl &view) {
    auto vars = view.uniqueVars;
    llvm::outs() << "TensorView for: " << view.getValue() << "\nWith vars: ";
    llvm::outs().flush();
    for (auto varIter: llvm::enumerate(vars)) {
        auto var = varIter.value();
        unsigned int index = varIter.index();
        os << var.getLogicalIndexVar();
        if (index < vars.size() - 1) {
            os << ", ";
        }
    }
    // os << std::endl;
    return os;
}

std::ostream &operator<<(std::ostream &os, const TensorView &view) {
    os << *view.tensorView;
    return os;
}

mlir::Value getViewValue(const TensorView &view) { return view.getValue(); }

void Table::dump(const std::shared_ptr<AnalysisScope> &scope, const std::vector<IndexVar> &vars) {
    using Row = tabulate::Table::Row_t;
    using RowStream = tabulate::RowStream;
    tabulate::Table table;
    const auto &tensors = scope->views;
    auto header = RowStream{};
    header << "";
    for (const auto &tensor: tensors) {
        //        header << scope->tensorNames[tensor.getValue()];
        header << scope->tensorViewNames[tensor];
    }
    table.add_row(header);
    for (const auto &var: vars) {
        auto newRow = RowStream{};
        newRow << var;
        for (const auto &tensor: tensors) {
            if (this->columns.count(tensor) && this->columns[tensor].count(var)) {
                newRow << this->columns.at(tensor)[var]->getLabel();
            } else {
                newRow << "Empty";
            }
        }
        table.add_row(newRow);
    }

    auto newRow = RowStream{};
    newRow << "Value";
    for (const auto &tensor: tensors) {
        if (this->columns[tensor].has_value()) {
            newRow << this->columns[tensor].getMutableValue()->getLabel();
        } else {
            newRow << "Empty";
        }
    }
    table.add_row(newRow);
    std::stringstream ss;
    ss << table;
    std::string line;
    while (std::getline(ss, line)) {
        std::cout << "// " << line << std::endl;
    }
}

// Printing function to print out all orders nicely
template<typename T>
void printAllOrders(std::vector<std::vector<T>> allOrders) {
    std::cout << "TOPOLOGICAL SORT RESULTS" << std::endl;
    for (int i = 0; i < allOrders.size(); i++) {
        auto order = allOrders.at(i);
        std::cout << "[" << i << "] ";
        for (auto node: order) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }

}
