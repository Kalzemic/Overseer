package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"strconv"
)

const (
	baseDir      = "/etc/overseer"
	projectsDir  = baseDir + "/projects"
	modulesDir   = baseDir + "/modules"
	
)


func getCgroupId(path string) (uint64, error){
	cgroup_path := filepath.Join("/sys/fs/cgroup",path)
	info, err := os.Stat(cgroup_path)
	if err != nil{
		return 0,err 
	}
	stat := info.Sys().(*syscall.Stat_t)
	return stat.Ino, nil 
}

func loadModules(modules string, cgroupID string) {
	for module := range strings.FieldsSeq(modules) {
		modulePath := filepath.Join(modulesDir, module)
		cmd := exec.Command("make", "-C", modulePath, "CGROUP="+cgroupID, "run")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "❌ Failed to load module %s: %v\n", module, err)
			os.Exit(1)
		}
	}
}




func createProject(name string, image string, modules string) {
    projectPath := filepath.Join(projectsDir, name)
    os.MkdirAll(projectPath, 0755)

    // pull image
    if err := exec.Command("docker", "pull", image).Run(); err != nil {
        fmt.Fprintf(os.Stderr, "Failed pulling docker image: %v\n", err)
        os.Exit(1)
    }

    fmt.Printf("✔ Project '%s' created with image '%s' and modules '%s'\n", name, image, modules)

    // docker create (non-blocking, capture container ID)
    createCmd := exec.Command("docker", "create",
        "--name", name,
        "--cgroup-parent=/overseer/"+name,
        "-it", image, "bash")

    containerIDBytes, err := createCmd.Output()
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to create container: %v\n", err)
        os.Exit(1)
    }
    containerID := strings.TrimSpace(string(containerIDBytes))

    // initial config (no cgroup yet, but include containerID)
    metadata := fmt.Sprintf("container=%s\nimage=%s\nmodules=%s\n", containerID, image, modules)
    if err := os.WriteFile(filepath.Join(projectPath, "config"), []byte(metadata), 0644); err != nil {
        fmt.Fprintf(os.Stderr, "Failed to write project config: %v\n", err)
        os.Exit(1)
    }

    // now start container interactively (blocking)
    startCmd := exec.Command("docker", "start", "-ai", name)
    startCmd.Stdin = os.Stdin
    startCmd.Stdout = os.Stdout
    startCmd.Stderr = os.Stderr
    if err := startCmd.Run(); err != nil {
        fmt.Fprintf(os.Stderr, "Failed to start container: %v\n", err)
        os.Exit(1)
    }

    // after container is started, resolve cgroup id
    cgroup_id, err := getCgroupId(fmt.Sprintf("/overseer/%s", name))
    if err != nil {
        fmt.Printf("Failed resolving cgroup id\n")
        os.Exit(1)
    }
    cgroupStr := strconv.FormatUint(cgroup_id, 10)

    // update config with cgroup id as well
    metadata = fmt.Sprintf("container=%s\ncgroup=%s\nimage=%s\nmodules=%s\n",
        containerID, cgroupStr, image, modules)
    if err := os.WriteFile(filepath.Join(projectPath, "config"), []byte(metadata), 0644); err != nil {
        fmt.Fprintf(os.Stderr, "Failed to update project config: %v\n", err)
        os.Exit(1)
    }

    // load modules
    loadModules(modules, cgroupStr)
}





func runProject(name string) {
	configPath := filepath.Join(projectsDir, name, "config")
	data, err := os.ReadFile(configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed searching for file")
		os.Exit(1)
	}
	config := make(map[string]string)
	for line := range strings.SplitSeq(string(data),"\n") {
		if kv := strings.SplitN(line, "=", 2); len(kv) == 2 {
			config[kv[0]] = kv[1]
		}
	}
	cgroup_id := config["cgroup"]
	image := config["image"]
	modules := config["modules"]
	

	fmt.Printf("🧠 Launching container with image: %s\n", image)
	fmt.Printf("📦 Modules assigned: %s\n", modules)


	cmd := exec.Command("docker", "start", "-ai", name)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to start container: %v\n", err)
		os.Exit(1)
	}
	loadModules(modules, cgroup_id)

}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: overseer <create|run> [args]")
		return
	}

	switch os.Args[1] {
	case "create":
		createCmd := flag.NewFlagSet("create", flag.ExitOnError)
		image := createCmd.String("image", "ubuntu", "defines image for project container")
		modules := createCmd.String("modules", "", "space seperated list of kernel modules")
		createCmd.Parse(os.Args[2:])
		args := createCmd.Args()
		if len(args) < 1 {
			fmt.Println("Usage: overseer create <project-name> --image <image> --modules <mod1 mod2 ...>")
			os.Exit(1)
		}
		name := args[0]
		createProject(name, *image, *modules)

	case "run":
		if len(os.Args) < 3 {
			fmt.Println("Usage: overseer run <project-name>")
			os.Exit(1)
		}
		projectName := os.Args[2]
		runProject(projectName)
	case "cluster":
		if len(os.Args) < 3 {
			fmt.Println("Usage: overseer cluster <create|run> ...")
			os.Exit(1)
		}

		switch os.Args[2] {
		case "create":
			createCmd := flag.NewFlagSet("create", flag.ExitOnError)
			image := createCmd.String("image", "ubuntu", "defines image for project container")
			modules := createCmd.String("modules", "", "kernel modules list")
			machinePath := createCmd.String("machinefile", "", "path to file containing list of ssh network ips")
			createCmd.Parse(os.Args[3:])
			args := createCmd.Args()
			if len(args) < 1 {
				fmt.Println("Usage: overseer create <project-name> --image <image> --modules <mod1 mod2 ...>")
				os.Exit(1)
			}
			name := args[0]
			createCluster(name, *image, *modules, *machinePath)
		case "run":
			if len(os.Args) < 4 {
				fmt.Println("Usage: overseer run <project-name>")
				os.Exit(1)
			}
			projectName := os.Args[3]
			runCluster(projectName)
		}
	default:
		fmt.Println("Unknown command:", os.Args[1])
		os.Exit(1)
	}

}
