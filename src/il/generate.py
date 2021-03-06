#!/usr/bin/env python

import sys

import il
from sile_types import types
from il import IlGenException
from freevars import freevars
from symbol_table import SymbolTable

## This file is the intermediate code generator. You likely will
## need to make changes in this file!

## Some portions of the code generator will be working but others
## will not be. You will need to fill it in.

def generate(ast):
    mods = il.Modules()
    generate_module(mods, 'main', ast)
    return mods

def generate_module(modules, name, ast):
    mod = il.Module(name)
    modules.add(mod)
    return IlGenerator.generate(modules, mod, ast)

class IlGenerator(object):

    @staticmethod
    def generate(modules, module, ast):
        self = IlGenerator(modules, module)
        self.stmts(self.function().new_block(), ast)
        return self.module

    def __init__(self, modules, module):
        self.modules = modules
        self.module = module
        self.functions = [self.module.new_function('main', types.Function([], types.UNIT))]
        self.symbols = SymbolTable()

    def function(self):
        return self.functions[-1]

    def push_function(self, name, type, params, freevars):
        # Modifying new function to take in function reference.
        fn = self.module.new_function(
            name, type, params=params, parent=self.function().ref(), freevars=freevars)
        self.functions.append(fn)
        return fn

    def pop_function(self):
        return self.functions.pop()

    def new_register(self, regtype):
        return self.function().new_register(regtype)

    def stmts(self, blk, n):
        self.symbols = self.symbols.push()
        for kid in n.children:
            blk = self.stmt(blk, kid)
        self.symbols = self.symbols.pop()
        return blk

    def stmt(self, blk, n):
        return self.dispatch_stmt(blk, n, {
            "import": self.import_action,
            "decl": self.decl_action,
            "assign": self.assign_action,
            "expr-stmt": self.expr_stmt_action,
            "print": self.print_action,
            "if": self.if_action,
            "while": self.while_action,
            "label": self.label_action,
            "break": self.break_action,
            "continue": self.continue_action,
            "function": self.function_action,
            "return": self.return_action,
            "stmts": self.stmts,
        })

    def import_action(self, blk, n):
        # Create a new module. 
        print("IMport")
        modules = generate(n.children[1])
        print("Done")
        for mod in modules:
            self.modules.add(mod)
        return blk
        
        
    def decl_action(self, blk, n):
        name = n.children[0].value
        dest = self.new_register(n.children[1].type)
        a, blk = self.expr(blk, dest, n.children[1])
        self.symbols[name] = a
        return blk

    def assign_action(self, blk, n):
        name = n.children[0].value
        dest = self.symbols[name]
        a, blk = self.expr(blk, dest, n.children[1])
        return blk

    def print_action(self, blk, n):
        a, blk = self.expr(blk, None, n.children[0])
        if isinstance(a, il.FunctionRef):
            closure = self.module.lookup(a).closure(self.module, self.symbols)
            if len(closure.captured) > 0:
                a, blk = self.create_closures(blk, closure)
        blk.append(il.Instruction(il.OPS['PRINT'], a, None, None))
        return blk

    def return_action(self, blk, n):
        a, blk = self.expr(blk, None, n.children[0])
        if isinstance(a, il.FunctionRef):
            closure = self.module.lookup(a).closure(self.module, self.symbols)
            if len(closure.captured) > 0:
                a, blk = self.create_closures(blk, closure)
        blk.append(il.Instruction(il.OPS['RTRN'], a, None, None))
        return blk

    def create_closures(self, blk, closure):
        ## Implements function closure creation
        registers = list()
        rewrite = dict()
        for name, operand in closure.captured.iteritems():
            if isinstance(operand, il.Closure):
                fn_ref = operand.fn
                operand, blk = self.create_closures(blk, operand)
                rewrite[fn_ref] = il.ClosureRegister(len(registers), fn_ref.type())
            registers.append(operand)
        result = self.new_register(closure.fn.type())
        rewrite[closure.fn] = il.ClosureRegister(len(registers), closure.fn.type())
        for idx, reg in enumerate(registers):
            rewrite[reg] = il.ClosureRegister(idx, reg.type())
        closure_code = self.rewrite(closure.fn, rewrite)
        blk.append(il.Instruction(
            il.OPS['CLOSURE'], closure_code.ref(), registers, result))
        return result, blk

    def rewrite(self, fn_ref, rewrites):
        ## Implements function rewriting for closure creation
        old = self.module.lookup(fn_ref)
        new = self.push_function(old.name + '-closure', old.type(), old.params, list())
        new.locals = old.locals
        def replace(operand):
            if isinstance(operand, il.FunctionRef) and operand in rewrites:
                return rewrites[operand]
            elif isinstance(operand, il.Register) and operand in rewrites:
                return rewrites[operand]
            elif isinstance(operand, il.Register) and operand.fn == old.ref():
                return il.Register(operand.id, new.ref(), operand.type())
            elif isinstance(operand, list):
                return [replace(inner) for inner in operand]
            return operand
        for old_blk in old.blocks:
            new_blk = new.new_block()
            for inst in old_blk.code:
                new_blk.append(il.Instruction(
                    inst.op, replace(inst.a), replace(inst.b),
                    replace(inst.result)))
        for idx, old_blk in enumerate(old.blocks):
            new_blk = new.blocks[idx]
            for link in old_blk.next:
                new_blk.link_to(new.blocks[link.target], link.link_type)
        self.pop_function()
        return new

    def if_action(self, entry, n):
        body = self.function().new_block()
        afterwards = self.function().new_block()
        
        # Evaluate the condition expr.
        cond, cond_out = self.expr(entry, None, n.children[0])
        
        # Evaluate the body of the conditional. 
        body_out = self.stmts(body, n.children[1])
        
        # Always goes to the afterwards block after if loop. 
        body_out.goto_link(afterwards)
        
        # If the cond is true, go to body else to afterwards. 
        cond_out.if_link(cond, body, afterwards)
        return afterwards

    def while_action(self, entry, n):
        header = self.function().new_block()
        body = self.function().new_block()
        afterwards = self.function().new_block()
        return self._while_action(entry, n, header, body, afterwards)

    def _while_action(self, entry, n, header, body, afterwards):
        self.function().push_loop(header, afterwards)
        entry.goto_link(header)
        cond, cond_out = self.expr(header, None, n.children[0])
        cond_out.if_link(cond, body, afterwards)
        body_out = self.stmts(body, n.children[1])
        body_out.goto_link(header)
        self.function().pop_loop()
        return afterwards

    def label_action(self, entry, n):
        # Blocks for continue, body and exit. 
        continue_blk = self.function().new_block()
        body_blk = self.function().new_block()
        exit_blk = self.function().new_block()
        
        # Add label to the symbol table mapping to a LabeledLoop. 
        label_name = n.children[0].value
        self.symbols[label_name] = il.LabeledLoop(continue_blk, exit_blk)
        
        # Use the internal while action to pass your own continue, body and exit_blks. 
        afterwards = self._while_action(entry, n.children[1], continue_blk, body_blk, exit_blk)
        return afterwards

    def break_action(self, blk, n):
        
        # If no children, exit out of the nearest loop. 
        if len(n.children) == 0:
            exit_blk = self.function().loop_exit()
        else:
            # Find the labelled loop record in the symbol_table. 
            # Retrieve the exit blk of the loop. 
            label = n.children[0].value
            exit_blk = self.symbols[label].exit_blk
        
        # Add a goto link to the exit block of the corresponding loop. 
        blk.goto_link(exit_blk)
        
        # Create a new empty block and return for other instructions to be added. 
        dead = self.function().new_block()
        return dead

    def continue_action(self, blk, n):
        if len(n.children) == 0:
            header = self.function().loop_cont()
        else:
            label = n.children[0].value
            header = self.symbols[label].continue_blk
        blk.goto_link(header)
        dead = self.function().new_block()
        return dead

    def expr_stmt_action(self, blk, n):
        _, blk = self.expr(blk, None, n.children[0])
        return blk

    def function_action(self, blk, n):
        function_type = n.children[2].value
        name = n.children[0].value
        body = n.children[3]
        params = [
            il.Param(id=idx, name=param.children[0].value, type=param.type)
            for idx, param in enumerate(n.children[1].children)]
        free = freevars(n)
        
        # Push the function into the function list. 
        # self.function() returns this function now. 
        fn = self.push_function(name, function_type, params, free)
        

        # Add function reference to the symbol table of the calling function. 
        self.symbols[fn.name] = fn.ref()
        
        # Push a new symbol table for the function. 
        self.symbols = self.symbols.push()
        
        # Create a new block for the function. 
        function_block = fn.new_block()
        for i, param in enumerate(params):
            
            # Create a local register for each parameter. 
            param_register = fn.new_register(param.type)
            
            # Add the parameter to symbol_table. 
            self.symbols[param.name] = param_register 
            
            # Append PRM instructions to the function_block. 
            # This instruction gives the id of each parameter which can be 
            # referenced by the called function to lookup the parameter value 
            # in the params of the function frame. 
            function_block.append(il.Instruction(il.OPS['PRM'], il.Constant(param.id, param.type), None, param_register))
            
        # Fill out instructions in the function_block. 
        self.stmts(function_block, body)
        
        # Remove the function from the function list. 
        # After this call, self.function() should not return this function. 
        self.pop_function()
        return blk

    def expr(self, blk, result, n):
        return self.dispatch_expr(blk, result, n, {
            "negate": self.negate,
            "+": self.binop(il.OPS['ADD']),
            "-": self.binop(il.OPS['SUB']),
            "*": self.binop(il.OPS['MUL']),
            "/": self.binop(il.OPS['DIV']),
            "%": self.binop(il.OPS['MOD']),         
            
            "==": self.binop(il.OPS['EQ']),
            "!=": self.binop(il.OPS['NE']),
            "<": self.binop(il.OPS['LT']),
            ">": self.binop(il.OPS['GT']),
            "<=": self.binop(il.OPS['LE']),
            ">=": self.binop(il.OPS['GE']),

            "not": self.not_op,
            "&&": self.and_op,
            "||": self.or_op,
            "call": self.call,
            "NAME": self.name,
            "INTEGER": self.number,
            "FLOAT": self.number,
        })

    def negate(self, blk, result, n):
        if result is None:
            result = self.new_register(n.type)
        a, blk = self.expr(blk, None, n.children[0])
        blk.append(il.Instruction(
            il.OPS['SUB'], il.Constant(0, n.type), a, result))
        return result, blk

    def binop(self, op):
        def binop(blk, result, n):
            if result is None:
                result = self.new_register(n.type)
            a, blk = self.expr(blk, None, n.children[0])
            b, blk = self.expr(blk, None, n.children[1])
            blk.append(il.Instruction(op, a, b, result))
            return result, blk
        return binop

    def not_op(self, blk, result, n):
        if not result:
            result = self.new_register(n.type)
        a, blk = self.expr(blk, result, n.children[0])
        blk.append(il.Instruction(il.OPS['NOT'], a, None, result))
        return result, blk

    def and_op(self, a_in_blk, result, n):
        if result is None:
            result = self.new_register(n.type)
        a, a_out_blk = self.expr(a_in_blk, result, n.children[0])
        
        b_in_blk = self.function().new_block()
        exit_blk = self.function().new_block()
        
        # If 1st condition true, go to second block, else exit block. 
        a_out_blk.if_link(a, b_in_blk, exit_blk) 
        #                     on-true    on-false 
        b, b_out_blk = self.expr(b_in_blk, result, n.children[1])
        b_out_blk.goto_link(exit_blk)
        return result, exit_blk

    def or_op(self, a_in_blk, result, n):
        if result is None:
            result = self.new_register(n.type)
        a, a_out_blk = self.expr(a_in_blk, result, n.children[0])
        b_in_blk = self.function().new_block()
        exit_blk = self.function().new_block()
        a_out_blk.if_link(a, exit_blk, b_in_blk)
        #                    on-true   on-false
        b, b_out_blk = self.expr(b_in_blk, result, n.children[1])
        b_out_blk.goto_link(exit_blk)
        return result, exit_blk

    def call(self, blk, result, n):
        function_name = n.children[0].value
        exprs = n.children[1]
        parameters = []
        
        # Resolve each parameter 
        for expr in exprs.children:
            r, _ = self.expr(blk, None, expr)
            parameters.append(r)
        
        if result is None:
            result = self.function().new_register(n.type)
            
        # SymbolTable returns the function reference. 
        blk.append(il.Instruction(il.OPS['CALL'], self.symbols.get(function_name), parameters, result))
        return result, blk

    def name(self, blk, result, n):
        if result is None:
            return self.symbols[n.value], blk
        blk.append(il.Instruction(
            il.OPS['MV'], self.symbols[n.value], None, result))
        return result, blk

    def number(self, blk, result, n):
        const = il.Constant(n.value, n.type)
        if result is None:
            return const, blk
        blk.append(il.Instruction(il.OPS['IMM'], const, None, result))
        return result, blk

    def dispatch_stmt(self, blk, n, labels_to_actions):
        for name, func in labels_to_actions.iteritems():
            if n.label == name:
                return func(blk, n)
        raise IlGenException(
            "got '{}', want one of: {}".format(n.label, labels_to_actions.keys()))

    def dispatch_expr(self, blk, result, n, labels_to_actions):
        for name, func in labels_to_actions.iteritems():
            if n.label == name:
                return func(blk, result, n)
        raise IlGenException(
            "got '{}', want one of: {}".format(n.label, labels_to_actions.keys()))

