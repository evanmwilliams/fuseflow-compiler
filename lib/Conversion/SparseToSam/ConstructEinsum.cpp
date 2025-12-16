//
// Created by rubensl on 11/3/24.
//
#include "ConstructEinsum.h"
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "FusedCIN.h"


class UnaryOpNode;

class BinaryOpNode;

class TensorNode;

class FuncOpNode;

void UnaryOpNode::accept(ASTVisitor &visitor) { visitor.visit(*this); }

void BinaryOpNode::accept(ASTVisitor &visitor) { visitor.visit(*this); }

void ScalarNode::accept(ASTVisitor &visitor) { visitor.visit(*this); }

void TensorNode::accept(ASTVisitor &visitor) { visitor.visit(*this); }

void FuncOpNode::accept(ASTVisitor &visitor) { visitor.visit(*this); }
