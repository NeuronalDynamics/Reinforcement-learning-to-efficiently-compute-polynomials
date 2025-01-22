-- src/Main.lean

import RLAgent

open RLAgent

def explore (s : State) (depth : Nat) : State :=
  if depth == 0 then s
  else
    let s' := naivePolicy s  -- pick next move
    explore s' (depth - 1)

def main : IO Unit := do
  let initial := State.mk (ArithmeticCircuit.add (ArithmeticCircuit.input "x1") (ArithmeticCircuit.input "x2"))
  let finalState := explore initial 10
  IO.println s!"Final circuit: {finalState.circuit}"
  IO.println s!"Final cost: {ArithmeticCircuit.cost finalState.circuit}"

-- Make `main` the entry point
@[default_entry_point]
def defaultMain := main
