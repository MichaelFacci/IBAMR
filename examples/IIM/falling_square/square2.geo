// Gmsh project created on Fri Dec 20 14:05:05 2024
SetFactory("OpenCASCADE");
//+
Ellipse(1) = {0, 0, 0, 0.05, 0.015, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Surface(1) = {1};
//+
Curve Loop(3) = {1};
//+
Surface(2) = {3};
//+
Disk(3) = {-1.2, 0.3, 0.2, 0.5, 0.25};
