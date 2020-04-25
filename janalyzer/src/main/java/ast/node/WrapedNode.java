package ast.node;

import com.alibaba.fastjson.annotation.JSONField;
import com.github.javaparser.ast.Node;

import java.util.ArrayList;

public class WrapedNode {
    private Node node;
    @JSONField(ordinal = 2)
    private String mask;
    @JSONField(ordinal = 1)
    private String nodeName;
    @JSONField(ordinal = 3)
    private ArrayList<WrapedNode> children = new ArrayList<>();

    public WrapedNode(){}

    public WrapedNode(Node node){
        this.node = node;
        if (node.getChildNodes().isEmpty()){
            setMask("1");
            setNodeName(node.toString());
        }else {
            setMask("0");
            String[] nameSplit = node.getClass().getName().split("\\.");
            if (nameSplit.length > 0){
                setNodeName(nameSplit[nameSplit.length-1]);
            }
        }
        for (Node child: node.getChildNodes()){
            addChild(child);
        }

    }

    public WrapedNode(String nodeName, String mask){
        setNodeName(nodeName);
        setMask(mask);
    }

    private void addChild(Node node) {
        WrapedNode wrapedNode = new WrapedNode(node);
        children.add(wrapedNode);
    }

    public Node getNode() {
        return node;
    }

    public void setNode(Node node) {
        this.node = node;
    }

    public String getMask() {
        return mask;
    }

    public void setMask(String mask) {
        this.mask = mask;
    }

    public String getNodeName() {
        return nodeName;
    }

    public void setNodeName(String nodeName) {
        this.nodeName = nodeName;
    }

    public ArrayList<WrapedNode> getChildren() {
        return children;
    }

    public void setChild(WrapedNode child) {
        this.children.add(child);
    }
}
