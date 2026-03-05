package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)


func loadClusterModules(modules, cgroupID, machine string) {
    for _, module := range strings.Fields(modules) {
        modulePath := filepath.Join(modulesDir, module)
        fmt.Printf("🔧 [%s] Loading module: %s (CGROUP=%s)\n", machine, module, cgroupID)

        cmd := exec.Command("ssh", machine, "make", "-C", modulePath, "CGROUP="+cgroupID, "run")
        cmd.Stdout = os.Stdout
        cmd.Stderr = os.Stderr

        if err := cmd.Run(); err != nil {
            fmt.Fprintf(os.Stderr, "❌ [%s] Failed to load module %s: %v\n", machine, module, err)
            os.Exit(1)
        }
    }
}

func createCluster(name, image, modules, hostpath string) {
    projectPath := filepath.Join(projectsDir, name)
    hosts, err := readHosts(hostpath)
    if err != nil {
        fmt.Fprintf(os.Stderr, "❌ Error reading machinefile: %v\n", err)
        return
    }

    if err := os.MkdirAll(projectPath, 0755); err != nil {
        fmt.Fprintf(os.Stderr, "❌ Failed to create local project dir: %v\n", err)
        return
    }

    machinePath := filepath.Join(projectPath, "machinefile")
    os.WriteFile(machinePath, []byte(strings.Join(hosts, "\n")), 0644)

    for _, ip := range hosts {
        // docker create
        createCmd := exec.Command("ssh", ip,
            "docker", "create",
            "--name", name,
            "--cgroup-parent=/overseer/"+name,
            "-it", image, "bash")
        out, err := createCmd.Output()
        if err != nil {
            fmt.Fprintf(os.Stderr, "❌ [%s] Failed to create container: %v\n", ip, err)
            continue
        }
        containerID := strings.TrimSpace(string(out))

        metadata := fmt.Sprintf("container=%s\nimage=%s\nmodules=%s\n", containerID, image, modules)
        remoteConfig := filepath.Join(projectPath, "config")

        cmd := exec.Command("ssh", ip, "bash", "-c", fmt.Sprintf("echo -e '%s' > %s", metadata, remoteConfig))
        cmd.Run()

        dest := fmt.Sprintf("%s:%s/machinefile", ip, projectPath)
        exec.Command("scp", machinePath, dest).Run()

        exec.Command("ssh", ip, "docker", "pull", image).Run()

        fmt.Printf("✅ Project '%s' defined on %s (container %s)\n", name, ip, containerID)
    }
}

func runCluster(name string) {
    projectPath := filepath.Join(projectsDir, name)
    configPath := filepath.Join(projectPath, "config")
    data, err := os.ReadFile(configPath)
    if err != nil {
        fmt.Fprintf(os.Stderr, "❌ Failed to read config: %v\n", err)
        os.Exit(1)
    }

    config := make(map[string]string)
    for _, line := range strings.Split(string(data), "\n") {
        if kv := strings.SplitN(line, "=", 2); len(kv) == 2 {
            config[kv[0]] = kv[1]
        }
    }

    modules := config["modules"]
    hosts, _ := readHosts(filepath.Join(projectPath, "machinefile"))

    for _, ip := range hosts {
        fmt.Printf("🚀 Starting container on %s...\n", ip)

        runCmd := exec.Command("ssh", ip, "docker", "start", "-ai", name)
        runCmd.Stdin = os.Stdin
        runCmd.Stdout = os.Stdout
        runCmd.Stderr = os.Stderr
        if err := runCmd.Run(); err != nil {
            fmt.Fprintf(os.Stderr, "❌ [%s] Failed to start container: %v\n", ip, err)
            continue
        }

        // resolve cgroup via ssh (your getCgroupId but remote)
        cgroupPath := fmt.Sprintf("/sys/fs/cgroup/overseer/%s", name)
        cgroupCmd := exec.Command("ssh", ip, "stat", "-c", "%i", cgroupPath)
        out, err := cgroupCmd.Output()
        if err != nil {
            fmt.Fprintf(os.Stderr, "❌ [%s] Failed to resolve cgroup id: %v\n", ip, err)
            continue
        }
        cgroupID := strings.TrimSpace(string(out))

        loadClusterModules(modules, cgroupID, ip)
    }
}


func readHosts(hostpath string)([]string,error){
	file, err := os.Open(hostpath)
	if err != nil{
		return []string{}, err 
	}

	var hosts []string 
	scanner := bufio.NewScanner(file)
	for scanner.Scan(){
		if line := strings.TrimSpace(scanner.Text()); line != "" {
			hosts = append(hosts,line)
		}
	}
	return hosts, scanner.Err()
}
