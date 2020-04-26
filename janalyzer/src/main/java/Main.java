import java.io.IOException;
import project.JZipProject;
public class Main {
    public static void main(String[] args) throws IOException{
        JZipProject jZipProject = new JZipProject(
                "D:\\coding/data/java_repos/",
                "D:\\coding/data/java_parsed/",
                "jdk.txt");
        jZipProject.parse(3143, jZipProject.getList().size(), false);
        //jZipProject.parse(0, 2, false);
    }
}
