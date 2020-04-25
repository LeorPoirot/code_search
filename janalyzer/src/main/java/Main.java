import java.io.IOException;
import project.JZipProject;
public class Main {
    public static void main(String[] args) throws IOException{
        JZipProject jZipProject = new JZipProject(
                "/Users/wenzan/projects/alibaba/data/java_repos/",
                "/Users/wenzan/projects/alibaba/data/",
                "jdk.txt");
        //jZipProject.parse(0, jZipProject.getList().size(), false);
        jZipProject.parse(0, 1, false);
    }
}
