#include "FusedCIN.h"
#include "Table.h"
#include "ortools/base/types.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <fstream>
#include <sstream>


// #include "ortools/graph/connected_components.h"
// #include "ortools/graph/dense_connected_component_finder.h"

namespace {

std::string orderToString(const std::vector<IndexVar> &order) {
    std::ostringstream os;
    for (size_t i = 0; i < order.size(); ++i) {
        if (i) os << ' ';
        os << "i" << order[i].getId();
    }
    return os.str();
}

std::string sanitizeOpName(mlir::Operation *op) {
    if (!op) return "unknown_op";
    auto sanitize = [](std::string s) {
        for (auto &c : s) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
        }
        return s.empty() ? std::string("op") : s;
    };
    // Symbol-named ops (e.g. func.func): use the symbol name.
    if (auto sym = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
        return sanitize(sym.getValue().str());
    }
    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    mlir::AsmState state(funcOp ? funcOp.getOperation() : op);
    std::string buf;
    llvm::raw_string_ostream rs(buf);
    if (op->getNumResults() > 0) {
        op->getResult(0).printAsOperand(rs, state);
    } else {
        rs << "op";
    }
    rs.flush();
    llvm::StringRef name(buf);
    if (name.starts_with("%")) name = name.drop_front();
    return sanitize(name.str());
}

mlir::LogicalResult runProgram(llvm::StringRef bin,
                               llvm::ArrayRef<llvm::StringRef> argv,
                               llvm::StringRef cwd, llvm::StringRef tag) {
    llvm::SmallString<256> savedCwd;
    if (auto ec = llvm::sys::fs::current_path(savedCwd)) {
        llvm::errs() << "failed to get cwd: " << ec.message() << "\n";
        return mlir::failure();
    }
    if (auto ec = llvm::sys::fs::set_current_path(cwd)) {
        llvm::errs() << "failed to chdir to '" << cwd << "': " << ec.message() << "\n";
        return mlir::failure();
    }
    std::string err;
    int rc = llvm::sys::ExecuteAndWait(bin, argv, /*Env=*/std::nullopt,
                                       /*Redirects=*/{}, /*SecondsToWait=*/0,
                                       /*MemoryLimit=*/0, &err);
    (void)llvm::sys::fs::set_current_path(savedCwd);
    if (rc != 0) {
        llvm::errs() << tag << " returned " << rc;
        if (!err.empty()) llvm::errs() << ": " << err;
        llvm::errs() << "\n";
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult invokeHoneybee(llvm::StringRef bin, llvm::StringRef libraryPath,
                                   llvm::StringRef progPath,
                                   llvm::StringRef transcriptPath,
                                   llvm::StringRef cwd) {
    auto resolved = llvm::sys::findProgramByName(bin);
    if (!resolved) {
        llvm::errs() << "honeybee binary not found: " << bin << "\n";
        return mlir::failure();
    }
    llvm::SmallVector<llvm::StringRef> argv;
    argv.push_back(*resolved);
    argv.push_back("interact");
    argv.push_back("--library");
    argv.push_back(libraryPath);
    argv.push_back("--out");
    argv.push_back(transcriptPath);
    argv.push_back(progPath);
    return runProgram(*resolved, argv, cwd, "honeybee");
}

mlir::LogicalResult executeTranscript(llvm::StringRef transcriptPath,
                                      llvm::StringRef cwd) {
    auto python = llvm::sys::findProgramByName("python3");
    if (!python) python = llvm::sys::findProgramByName("python");
    if (!python) {
        llvm::errs() << "python3/python not found on PATH\n";
        return mlir::failure();
    }
    llvm::SmallVector<llvm::StringRef> argv;
    argv.push_back(*python);
    argv.push_back(transcriptPath);
    return runProgram(*python, argv, cwd, "python");
}

mlir::LogicalResult findLoopOrderJson(llvm::StringRef sessionDir,
                                      llvm::SmallString<256> &out) {
    llvm::SmallString<256> outputDir(sessionDir);
    llvm::sys::path::append(outputDir, "output");
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator dir(outputDir, ec), end;
         dir != end && !ec; dir.increment(ec)) {
        llvm::SmallString<256> candidate(dir->path());
        llvm::sys::path::append(candidate, "loop_order.json");
        if (llvm::sys::fs::exists(candidate)) {
            out = candidate;
            return mlir::success();
        }
    }
    llvm::errs() << "no loop_order.json found under '" << outputDir << "'\n";
    return mlir::failure();
}

mlir::LogicalResult writeProgToml(llvm::StringRef path,
                                  llvm::StringRef opName,
                                  int numLoops,
                                  const std::vector<std::vector<IndexVar>> &allOrders) {
    std::ofstream out(path.str());
    if (!out) {
        llvm::errs() << "failed to open '" << path << "' for writing\n";
        return mlir::failure();
    }
    out << "[[Prop]]\n"
        << "name = \"P_Op\"\n"
        << "[Prop.args]\n"
        << "op_name = \"" << opName.str() << "\"\n"
        << "num_loops = " << numLoops << "\n\n";
    for (const auto &order : allOrders) {
        out << "[[Prop]]\n"
            << "name = \"P_LoopOrderOption\"\n"
            << "[Prop.args]\n"
            << "op_name = \"" << opName.str() << "\"\n"
            << "order = \"" << orderToString(order) << "\"\n\n";
    }
    out << "[Goal]\n"
        << "name = \"DataflowOrder\"\n"
        << "args = {}\n";
    return mlir::success();
}

// Pick the chosen order index by matching the JSON's "order" string against
// our orderToString of each candidate.
int selectViaHoneybee(mlir::Operation *op,
                      const std::vector<std::vector<IndexVar>> &allOrders,
                      llvm::StringRef bin, llvm::StringRef library,
                      llvm::StringRef outputDir) {
    std::string opName = sanitizeOpName(op);
    llvm::SmallString<256> sessionDir(outputDir);
    llvm::sys::path::append(sessionDir, opName);
    if (auto ec = llvm::sys::fs::create_directories(sessionDir)) {
        llvm::errs() << "failed to create session dir '" << sessionDir
                     << "': " << ec.message() << "\n";
        return 0;
    }
    llvm::SmallString<256> progPath(sessionDir);
    llvm::sys::path::append(progPath, "prog.hb.toml");
    llvm::SmallString<256> transcriptPath(sessionDir);
    llvm::sys::path::append(transcriptPath, "session.py");

    int numLoops = allOrders.empty() ? 0 : static_cast<int>(allOrders.front().size());
    llvm::outs() << "\n=== Honeybee loop-order selection for '" << opName
                 << "' (" << allOrders.size() << " valid orders) ===\n";
    if (mlir::failed(writeProgToml(progPath.str(), opName, numLoops, allOrders))) return 0;
    if (mlir::failed(invokeHoneybee(bin, library, progPath, transcriptPath, sessionDir))) return 0;
    if (mlir::failed(executeTranscript(transcriptPath, sessionDir))) return 0;

    llvm::SmallString<256> jsonPath;
    if (mlir::failed(findLoopOrderJson(sessionDir, jsonPath))) return 0;
    auto bufOrErr = llvm::MemoryBuffer::getFile(jsonPath);
    if (!bufOrErr) {
        llvm::errs() << "failed to read " << jsonPath << "\n";
        return 0;
    }
    auto parsed = llvm::json::parse(bufOrErr.get()->getBuffer());
    if (!parsed) {
        llvm::errs() << "failed to parse JSON at " << jsonPath << "\n";
        return 0;
    }
    auto *obj = parsed->getAsObject();
    if (!obj) return 0;
    auto chosen = obj->getString("order");
    if (!chosen) {
        llvm::errs() << "no \"order\" field in " << jsonPath << "\n";
        return 0;
    }
    std::string chosenStr = chosen->str();
    for (size_t i = 0; i < allOrders.size(); ++i) {
        if (orderToString(allOrders[i]) == chosenStr) return static_cast<int>(i);
    }
    llvm::errs() << "chosen order '" << chosenStr
                 << "' did not match any valid order; defaulting to [0]\n";
    return 0;
}

} // namespace

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

std::vector<IndexVar> AnalysisScope::getLoopOrder(mlir::Operation *op) {
    std::vector<IndexVar> resultOrder;
    auto allOrdersIndexVars = getAllLoopOrders();

    std::cout << "// # of possible loop orders: " << allOrdersIndexVars.size() << std::endl;

    // FIXME (owhsu): Currently just get first result of all valid orders
    int selectOrder = 0;
    if (useHoneybee && op && !allOrdersIndexVars.empty()) {
        selectOrder = selectViaHoneybee(op, allOrdersIndexVars, honeybeeBinary,
                                        honeybeeLibrary, honeybeeOutputDir);
    } else if (useUserInput) {
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
