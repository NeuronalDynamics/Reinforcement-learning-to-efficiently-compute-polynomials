-- src/Polynomials.lean

namespace Polynomials

abbrev VarName := String  -- a variable name, e.g. "x1", "x2", ...

inductive Poly
  | var (v : VarName)
  | const (c : Int)          -- or Real, if you prefer
  | add  (p1 p2 : Poly)
  | mul  (p1 p2 : Poly)
  deriving Repr, Inhabited

open Poly

def eval (assignment : VarName → Int) : Poly → Int
  | var v     => assignment v
  | const c   => c
  | add p1 p2 => eval assignment p1 + eval assignment p2
  | mul p1 p2 => eval assignment p1 * eval assignment p2

end Polynomials
