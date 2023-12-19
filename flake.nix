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
    in
    {
      formatter.${system} = pkgs.nixpkgs-fmt;
      packages.${system}.default = naersk'.buildPackage {
        nativeBuildInputs = with pkgs; [ ocl-icd ];
        src = ./.;
      };
      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [ rustc cargo ocl-icd ];
        shellHook = ''
          export OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors
        '';
      };
    };
}
