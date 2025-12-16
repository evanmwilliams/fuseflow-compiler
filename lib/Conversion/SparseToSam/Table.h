#ifndef TABLE_H
#define TABLE_H

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <variant>
#include "Vars.h"
#include "mlir/IR/Value.h"
#include "tabulate.h"

class FusedCIN;

class TensorView;

class UniqueVar;

class IndexVar;

class AnalysisScope;

class TensorViewImpl;

class Table;

// Utility debug function
mlir::Value getViewValue(const TensorView &view);

// class Once;
class Column;

using RefCrd = std::pair<std::optional<mlir::Value>, mlir::Value>;

template<typename T>
class Once {
public:
    explicit Once(T value) : cell(value) {}

    explicit Once(std::function<T(Table &)> func) : cell(func) {}

    T evaluate(Table &table) {
        if (std::holds_alternative<T>(this->cell)) {
            return std::get<T>(this->cell);
        }
        auto val = std::get<std::function<T(Table &)>>(this->cell)(table);
        this->cell = val;
        return val;
    }

    unsigned int value_type() { return this->cell.index(); }

    // friend static std::shared_ptr<Once> make_shared(std::function<T(Table&)> func);

private:
    // Each cell in a table is either a lambda for computing the value or the
    // actual value (either a ref/crd pair or val/crd pair)
    std::variant<T, std::function<T(Table &)>, std::monostate> cell;
};

template<typename T>
static std::shared_ptr<Once<T>> make_once(std::function<T(Table &)> func) {
    auto once = std::make_shared<Once<T>>(func);
    // std::cout << "Inside once: " << once->value_type() << std::endl;
    return once;
}

template<typename T>
static std::shared_ptr<Once<T>> make_once(T val) {
    auto once = std::make_shared<Once<T>>(val);
    // std::cout << "Inside once: " << once->value_type() << std::endl;
    return once;
}

template<typename T>
class Cell {
public:
    virtual ~Cell() = default;

    virtual T get(Table &table) = 0;

    virtual std::string getLabel() = 0;

    virtual void check() = 0;
};

template<typename T, typename U>
class CellImpl final : public Cell<T> {
public:
    explicit CellImpl(std::shared_ptr<Once<U>> cell, std::function<T(U)> convert, const std::string &cellLabel = "") :
            conversion(convert), cell(cell), label(cellLabel) {
    }

    T get(Table &table) override {
        auto val = this->conversion(this->cell->evaluate(table));
        return val;
    }

    std::string getLabel() override { return this->label; }

    // T get(Table& table, std::function<T(U)> convertFunc)
    // {
    // auto val = convertFunc(this->cell->evaluate(table));
    // return val;
    // }

    void check() override { std::cout << "Inside cell: " << this->cell->value_type() << std::endl; }

private:
    std::function<T(U)> conversion;
    std::shared_ptr<Once<U>> cell;
    std::string label;
};

template<typename T, typename U>
static std::shared_ptr<Cell<T>> make_cell(std::function<U(Table &)> func) {
    return std::make_shared<CellImpl<T, U>>(make_once(func));
}

template<typename T, typename U>
static std::unique_ptr<Cell<T>> make_unique_cell(std::function<U(Table &)> func, std::string label = "") {
    // Default conversion function just returns U as T returning RefCrd pair
    std::function convFunc = [=](U val) -> T { return val; };
    auto once = make_once(func);
    auto cell = std::make_unique<CellImpl<T, U>>(once, convFunc, label);
    return cell;
}

template<typename T, typename U>
static std::unique_ptr<Cell<T>> make_unique_cell(std::function<U(Table &)> func, std::function<T(U)> convFunc,
                                                 std::string label = "") {
    auto once = make_once(func);
    auto cell = std::make_unique<CellImpl<T, U>>(once, convFunc, label);
    return cell;
}

template<typename T, typename U>
static std::unique_ptr<Cell<T>> make_unique_cell(const std::shared_ptr<Once<U>> &once, std::function<T(U)> convFunc,
                                                 std::string label = "") {
    auto cell = std::make_unique<CellImpl<T, U>>(once, convFunc, label);
    return cell;
}

template<typename T, typename U>
static std::unique_ptr<Cell<T>> make_unique_cell(const std::shared_ptr<Once<U>> &once, std::string label = "") {

    std::function convFunc = [=](U val) -> T { return val; };
    auto cell = std::make_unique<CellImpl<T, U>>(once, convFunc, label);
    return cell;
}

static std::string INIT;

class Column {
public:
    std::pair<std::optional<mlir::Value>, mlir::Value>
    at(const IndexVar &dim, Table &table, const mlir::Value tensorVal = mlir::Value(), std::string &t = INIT) {
        if (this->indices.find(dim) == this->indices.end()) {
            std::cout << "No entry found for dim: " << dim << std::endl;
            llvm::outs() << "From tensor: " << tensorVal << "\n";
            exit(1);
        }
        t = this->indices[dim]->getLabel();
        auto val = this->indices[dim]->get(table);
        return val;
    }

    std::unique_ptr<Cell<std::pair<std::optional<mlir::Value>, mlir::Value>>> &operator[](const IndexVar &dim) {
        return this->indices[dim];
    }

    // TODO: Add assign operator

    std::unique_ptr<Cell<mlir::Value>> &getMutableValue() { return this->value; }

    [[nodiscard]] bool has_value() const { return this->value != nullptr; }


    mlir::Value getValue(Table &table, std::string &t = INIT) const {
        t = this->value->getLabel();
        return this->value->get(table);
    }

    void setValue(std::unique_ptr<Cell<mlir::Value>> val) { this->value = std::move(val); }

    [[nodiscard]] unsigned int count(const IndexVar &var) const { return indices.count(var); }

    [[nodiscard]] bool empty() const { return indices.empty(); }

    //    Column& operator=(const Column& col) {
    //    }

private:
    // Each column in a table represents all the cells for a tensorview, where the
    // cell can either hold a ref/crd pair or just a crd with the ref being
    // represented as None (spacc case)
    std::map<IndexVar, std::unique_ptr<Cell<std::pair<std::optional<mlir::Value>, mlir::Value>>>, IndexVarCompare>
            indices;
    std::unique_ptr<Cell<mlir::Value>> value;
};

class Table {
public:
    Table() : columns(std::map<TensorView, Column, TensorViewCompare>()) {}

    void append(const TensorView &view, Column &column) { this->columns[view] = std::move(column); }

    void extend(Table &rhs) { this->columns.merge(rhs.columns); }

    RefCrd at(const TensorView &view, const IndexVar &var) {
        std::string label;
//        std::cout << "Getting index: " << this->columns.at(view)[var]->getLabel() << std::endl;
        auto val = this->columns.at(view).at(var, *this, getViewValue(view), label);
        path.push_back(label);
        return val;
    }

    Column &operator[](const TensorView &view) { return this->columns[view]; }

    mlir::Value getValue(const TensorView &view) {
        std::string label;
//        std::cout << "Retrieving value: " << this->columns.at(view).getMutableValue()->getLabel() << std::endl;
        const auto val = this->columns.at(view).getValue(*this, label);
        path.push_back(label);
        return val;
    }

    void dump(const std::shared_ptr<AnalysisScope> &scope, const std::vector<IndexVar> &vars);

    void dumpPath() {
        for (const auto &p: path) {
            std::cout << "Path: " << p << std::endl;
        }
        std::cout << std::endl << std::endl;
    }

private:
    std::map<TensorView, Column, TensorViewCompare> columns;
    std::vector<std::string> path;
};

#endif // TABLE_H
