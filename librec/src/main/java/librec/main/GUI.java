package librec.main;

import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JButton;
import java.awt.BorderLayout;
import javax.swing.JList;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import javax.swing.JTextField;
import javax.swing.JTextArea;
import java.awt.Color;
import java.awt.SystemColor;
import javax.swing.ListSelectionModel;
import javax.swing.event.ListSelectionListener;
import javax.swing.event.ListSelectionEvent;
import javax.swing.JComboBox;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JSpinner;
import javax.swing.JLabel;
import javax.swing.SwingConstants;
import javax.swing.SpinnerNumberModel;
import javax.swing.JScrollBar;
import javax.swing.JSlider;

public class GUI {

	private JFrame frame;
	private JTextField GroupRatings;
	private JTextField UserRatingsField;
	private JTextField PredictionsField;
	private JTextField GroupIdField;
	private JTextField UserInfoField;
	private JTextField textField;

	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					GUI window = new GUI();
					window.frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the application.
	 */
	public GUI() {
		initialize();
	}

	/**
	 * Initialize the contents of the frame.
	 */
	private void initialize() {
		frame = new JFrame();
		frame.getContentPane().setBackground(SystemColor.menu);
		frame.setBounds(100, 100, 514, 351);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(null);
		frame.setTitle("Group Recommeder System");
		
		JButton btnRun = new JButton("Run");
		btnRun.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				LibRec.main(null);
			}
		});
		btnRun.setBounds(0, 278, 244, 25);
		frame.getContentPane().add(btnRun);
		
		JButton btnNewButton = new JButton("Open Group Ratings");
		btnNewButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
			}
		});
		btnNewButton.setBounds(1, 33, 146, 25);
		frame.getContentPane().add(btnNewButton);
		
		GroupRatings = new JTextField();
		GroupRatings.setBounds(148, 33, 347, 25);
		frame.getContentPane().add(GroupRatings);
		GroupRatings.setColumns(10);
		
		JButton BtnUserRatings = new JButton("Open User Ratings");
		BtnUserRatings.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
			}
		});
		BtnUserRatings.setBounds(1, 63, 146, 25);
		frame.getContentPane().add(BtnUserRatings);
		
		UserRatingsField = new JTextField();
		UserRatingsField.setColumns(10);
		UserRatingsField.setBounds(148, 63, 347, 25);
		frame.getContentPane().add(UserRatingsField);
		
		JButton BtnPredictions = new JButton("Open User Predict");
		BtnPredictions.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
			}
		});
		BtnPredictions.setBounds(1, 93, 146, 25);
		frame.getContentPane().add(BtnPredictions);
		
		PredictionsField = new JTextField();
		PredictionsField.setColumns(10);
		PredictionsField.setBounds(148, 93, 347, 25);
		frame.getContentPane().add(PredictionsField);
		
		JButton BtnGroupId = new JButton("Open GroupId");
		BtnGroupId.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
			}
		});
		BtnGroupId.setBounds(1, 123, 146, 25);
		frame.getContentPane().add(BtnGroupId);
		
		GroupIdField = new JTextField();
		GroupIdField.setColumns(10);
		GroupIdField.setBounds(148, 123, 347, 25);
		frame.getContentPane().add(GroupIdField);
		
		JComboBox Methods = new JComboBox();
		Methods.setModel(new DefaultComboBoxModel(new String[] {"Average", "Average Measury", "Least Measury", 
				"Most Measury", "Most Popular", "Add", "Multiplicative", "Border Copeland"}));
		Methods.setBounds(1, 208, 146, 25);
		frame.getContentPane().add(Methods);
		
		UserInfoField = new JTextField();
		UserInfoField.setColumns(10);
		UserInfoField.setBounds(148, 153, 347, 25);
		frame.getContentPane().add(UserInfoField);
		
		JButton BtnUserInfo = new JButton("Open User Info");
		BtnUserInfo.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
			}
		});
		BtnUserInfo.setBounds(1, 153, 146, 25);
		frame.getContentPane().add(BtnUserInfo);
		
		JLabel lblNewLabel = new JLabel("Thresshold:");
		lblNewLabel.setBounds(1, 240, 90, 25);
		frame.getContentPane().add(lblNewLabel);
		
		JLabel lblWelcomeToGroup = new JLabel("Welcome To Group Recommeder System");
		lblWelcomeToGroup.setHorizontalAlignment(SwingConstants.CENTER);
		lblWelcomeToGroup.setBounds(1, 5, 490, 25);
		frame.getContentPane().add(lblWelcomeToGroup);
		
		textField = new JTextField();
		textField.setBounds(85, 240, 62, 25);
		frame.getContentPane().add(textField);
		textField.setColumns(10);
		
		JButton SetRun = new JButton("Set Run");
		SetRun.setBounds(245, 278, 251, 25);
		frame.getContentPane().add(SetRun);
		
		JComboBox comboBox = new JComboBox();
		comboBox.setModel(new DefaultComboBoxModel(new String[] {"Add", "Multiplicative", "Border Copeland"}));
		comboBox.setBounds(180, 208, 146, 25);
		frame.getContentPane().add(comboBox);
		
		JLabel lblNewLabel_1 = new JLabel("Rating");
		lblNewLabel_1.setBounds(1, 180, 56, 25);
		frame.getContentPane().add(lblNewLabel_1);
		
		JLabel lblRanking = new JLabel("Ranking");
		lblRanking.setBounds(180, 184, 56, 25);
		frame.getContentPane().add(lblRanking);
		
		
	}
}
