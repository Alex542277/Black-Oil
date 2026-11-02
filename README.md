# Black-Oil
Numerical implementation of a fractional-order Black-Oil model for two-phase oil-water flow. Features a Caputo time-fractional derivative, IMPES coupling, fractional Euler scheme, and validation against Buckley-Leverett solutions.
# Fractional-Order Modeling of Saturation Dynamics in Black-Oil Reservoir Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LaTeX](https://img.shields.io/badge/LaTeX-Manuscript-blue)](https://www.latex-project.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)

## 📖 Overview

This repository contains the numerical implementation and the LaTeX manuscript associated with the paper:

> **Fractional-Order Modeling of Saturation Dynamics in Black-Oil Reservoir Simulation**
> *Jamal Adetola, Alexandre L. Dahoue, Ganiou Atindehou, Koffi Wilfrid Houédanou, and Mohamed Allaoui*

The project investigates the numerical simulation of incompressible two-phase oil–water flow in porous media. A Caputo fractional time derivative is introduced into the water-saturation equation to capture memory effects characteristic of heterogeneous and fractured reservoirs. The pressure–saturation coupling is handled using an IMPES-type strategy, and the fractional equation is discretized with a fractional Euler (product-integration) scheme.

## 🔬 Key Features

- **Physical Model:** Incompressible two-phase oil–water flow (Black-Oil type).
- **Fractional Calculus:** Caputo time-fractional derivative of order \(\alpha \in (0,1)\).
- **Numerical Scheme:** Fractional Euler method (product integration) with IMPES coupling.
- **Well Model:** Simplified Peaceman-type source/sink terms.
- **Validation:** Comparison with the analytical Buckley–Leverett solution.
- **Sensitivity Analysis:** Accuracy, time-step sensitivity, and computational cost.

## 📂 Repository Structure
