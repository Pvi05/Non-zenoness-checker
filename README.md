# Non-zenoness checker

Implementation of an algorithm checking for sufficient conditions of non-zenoness in a UPPAAL system.

## Motivation

When proving programs and using timed automata, non-zenoness is a crucial property to ensure realistic behavior for our models.
An automaton (which models a program or a system) is zeno when there is the possibility for infinite actions in finite time, i.e., the automaton loops infinitely rapidly between different states.
Currently, UPPAAL, the mainstream program for timed automata model checking, doesn't have a feature to prove the non-zenoness property of an automaton. Non-zenoness is indeed a delicate property to address, especially when using synchronisation between multiple automata.

Our algorithm uses a sufficient condition for single-automaton systems which is extended to multiple-automata synchronisation by checking couples of loops in relation through one synchronisation. It does not provide a sufficient and necessary condition, but it does highlight loops at risk of being zeno, affirming the non-zenoness of a system if no such loops are found.

This program was done as a side part of a different project regarding checking properties of an aircraft TCAS system; and was completed alone in one day as an algorithmic challenge.

The program uses UPPAAL XML saves for a system and performs the following tasks:

• A cycle-detection algorithm is performed on the system to discover all local structural (control) loops.

• A second stage determines which loops are non-zeno that satisfy the sufficient condition. A loop is sufficiently non-zeno if there exists a clock such that: it is reset to 0 at one point and the clock needs to be above a certain value c > 0 for all guards and invariants to be satisfied.

Intuitively, this condition ensures the loops cannot be triggered infinitely while the clock is at zero or close to zero.

• The local loops are matched according to their half-actions (regarding synchronized actions), to produce a couple list alongside a second list of loops without synchronisation.

• Both lists are checked to return potentially unsafe loops, including couples of loops that are synchronized between them or a single loop. A couple is potentially unsafe if no loop in it is strongly non-zeno (the sufficient condition is not satisfied); similarly, a non-synchronising loop is potentially unsafe if it is not strongly non-zeno.

## Usage

The file requires ElementTree, Lark, re and itertools to function.
To perform a check on an XML file, call the program followed by the file location:

```bash
python3 Zeno_checker.py system.xml
```


As a one-day challenge, it doesn't address all UPPAAL functionalities. Mainly, broadcast channels are not supported, and global clocks might not be properly taken into account (prefer declaring them locally).
