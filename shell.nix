{ pkgs ? import <nixpkgs> {} }:
let
  myTarball = builtins.fetchTarball {
    url = "https://nixos.org/channels/nixos-unstable/nixexprs.tar.xz";
    sha256="sha256:1wj08lxiqzn78fa10d4sn3bdwl28mfarmr4m6q5frf1nab8a6rg4";
  };
  unstable = import myTarball { };

in
pkgs.mkShell {
   name = "cuda-env-shell";
   packages = [
     (unstable.python313.withPackages (python-pkgs: with python-pkgs; [
	#torchWithCuda
	torch
    numpy
	transformers
	safetensors
	datasets
     ]))
   ];

}
