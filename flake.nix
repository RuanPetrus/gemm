{
  description = "Cuda c++ enviroment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    pkgs = nixpkgs.legacyPackages."x86_64-linux";
  in {
    devShells."x86_64-linux".default = pkgs.mkShell {
      buildInputs = with pkgs; [
        python311
        python311Packages.numpy
        gcc12
        gdb
        pkg-config
        cudatoolkit
        git
        autoconf
        gnumake
        util-linux
        gperf
        unzip
        linuxPackages.nvidia_x11
        binutils
        cudaPackages.cuda_nvrtc
        cudaPackages.cuda_cudart
      ];
      shellHook = ''
        export CUDA_PATH=${pkgs.cudatoolkit}
        export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
      '';
    };
  };
}
