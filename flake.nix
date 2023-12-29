{
  inputs = {
    naersk.url = "github:nix-community/naersk";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
  };

  outputs = { self, naersk, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = (import nixpkgs) {
        inherit system;
      };
      naersk' = pkgs.callPackage naersk { };
      buildInputs = with pkgs; [ 
          # Build
          cmake
          extra-cmake-modules
          pkg-config

          # OpenCL / OpenGL
          ocl-icd
          libGL
          
          # x11 features
          xorg.libX11
          xorg.libXrandr
          xorg.libXinerama
          xorg.libXcursor
          xorg.libXi

          # Wayland features
          wayland
          wayland-protocols 
          libxkbcommon

          # Other
          udev
          alsa-lib
          vulkan-loader
      ];
    in
    {
      formatter.${system} = pkgs.nixpkgs-fmt;
      packages.${system}.default = naersk'.buildPackage rec {
        inherit buildInputs;
        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}";
        src = ./.;
      };
      devShells.${system}.default = pkgs.mkShell rec {
        nativeBuildInputs = with pkgs; [ 
          rustup 
          cargo 
          clinfo
        ];
        inherit buildInputs;
        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}";
        GLFW_LIB_DIR = "${pkgs.glfw}/lib";
      };
    };
}
